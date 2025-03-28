import torch
import numpy as np
import collections
import os
import time
import glob
import multiprocessing
from utils.parser import setup_parser
from utils.loss import MELoss, GaussianNLL, LaplacianNLL, WeightedGaussianNLL, WeightedLaplacianNLL
from utils.transforms import denormalize
from models.models import Net
from dataset import GEDIDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import trange

try:
    import wandb
except ImportError:
    print("Install wandb to log to Weights & Biases")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    


def train_epoch(model, error_metrics, train_loader, optimizer, epoch, writer, args):
    # init running error
    training_metrics = collections.defaultdict(lambda: 0)

    model.train()
    running_loss = 0
    max_iter = args.n_iter if args.n_iter is not None else len(train_loader)
    
    start = time.time()
    total_step = len(train_loader)
    for i, (images, labels, weights) in enumerate(train_loader):        
        images, labels, weights= images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32), weights.to(device, dtype=torch.float32)
        predictions = model(images).squeeze()

        if args.downsample:
            M, N = labels.shape[1:]
            K = 3
            MK = M//K
            NK = N//K
            labels = torch.nanmean(labels.reshape(labels.shape[0], MK, K, NK, K), axis=(1, 3))
            weights = torch.amin(weights.reshape(labels.shape[0], MK, K, NK, K), axis=(1, 3))
        mask = torch.isnan(labels)
        
        if args.num_outputs==2:
            predictions, log_variances = predictions[:,0], predictions[:,1]
            predictions, log_variances = predictions[~mask], log_variances[~mask]
            labels, weights = labels[~mask], weights[~mask]
            if args.loss_key in ("WeightedGaussianNLL", "WeightedLaplacianNLL"):
                loss = error_metrics['loss'](predictions, log_variances, labels, weights)
            else:
                loss = error_metrics['loss'](predictions, log_variances, labels)
        else:
            predictions = predictions[:, 0]  # fcn_6_gaussian always outputs mean & std
            predictions = predictions[~mask]
            labels, weights = labels[~mask], weights[~mask]
            loss = error_metrics['loss'](predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i%1000==0:
            with open(args.results_file, 'a') as f:
                f.write(f'Loss for batch {i}/{total_step} of epoch {epoch} is {loss} | Time is: {np.round(time.time() - start,2)}s \n')
        
        # denormalized labels and predictions
        predictions_ = denormalize(predictions, args.mean_agbd, args.sigma_agbd)
        predictions_ = torch.clamp(predictions_, 0) # preds should be >=0
        labels_ = denormalize(labels, args.mean_agbd, args.sigma_agbd)
        
        # compute metrics on every batch and add to running sum
        for metric in error_metrics:
            if metric == "loss":
                training_metrics[metric] = running_loss
            elif args.num_outputs == 2 and metric in ['GaussianNLL', 'LaplacianNLL']:
                training_metrics[metric] += error_metrics[metric](predictions, log_variances, labels).item()
            else:
                training_metrics[metric] += error_metrics[metric](predictions_, labels_).item()
                
        # test
        if i == max_iter:
            break
        
    for metric in error_metrics.keys():
        writer.add_scalar(f'training/{metric}', training_metrics[metric] / max_iter, global_step=epoch) 

    return np.around(running_loss / max_iter, decimals=3)


def test_epoch(model, error_metrics, test_loader, epoch, writer, args):
    # init running error
    validation_metrics = collections.defaultdict(lambda: 0)
    
    model.eval()
    max_iter = int(args.n_iter*0.1) if args.n_iter is not None else len(test_loader)
    
    print('max iter test: ' + str(max_iter))
    print('len test loader: ' + str(len(test_loader)))

    with torch.no_grad():
        for i, (images, labels, weights) in enumerate(test_loader):
            images, labels, weights = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32), weights.to(device, dtype=torch.float32)
            if args.downsample:
                M, N = labels.shape[1:]
                K = 3
                MK = M//K
                NK = N//K
                labels = torch.nanmean(labels.reshape(labels.shape[0], MK, K, NK, K), axis=(1, 3))
                weights = torch.amin(weights.reshape(labels.shape[0], MK, K, NK, K), axis=(1, 3))
            mask = torch.isnan(labels)
            predictions = model(images).squeeze()
            if args.num_outputs==2:
                predictions, log_variances = predictions[:,0], predictions[:,1]
                log_variances = log_variances[~mask]
            else:
                predictions = predictions[:, 0]
                
            predictions = predictions[~mask]
            labels, weights = labels[~mask], weights[~mask]
            
            # denormalized labels and predictions
            predictions_ = denormalize(predictions, args.mean_agbd, args.sigma_agbd)
            labels_ = denormalize(labels, args.mean_agbd, args.sigma_agbd)
            predictions_ = torch.clamp(predictions_, 0) # preds should be >=0
            
            for metric in error_metrics:
                if metric == "loss" and args.loss_key in ("WeightedGaussianNLL", "WeightedLaplacianNLL"):
                    validation_metrics[metric] += error_metrics[metric](predictions, log_variances, labels, weights).item()
                elif args.num_outputs == 2 and metric in ['GaussianNLL', 'LaplacianNLL', 'loss']:
                    validation_metrics[metric] += error_metrics[metric](predictions, log_variances, labels).item()
                else:
                    validation_metrics[metric] += error_metrics[metric](predictions_, labels_).item()
                
            if i == max_iter:
                break

        for metric in error_metrics.keys():
            writer.add_scalar(f'evaluation/{metric}', validation_metrics[metric] / max_iter, global_step=epoch)  
            with open(args.results_file, 'a') as f:
                f.write(f'{metric} {np.around(validation_metrics[metric] / max_iter, 3)} for step {epoch}\n')
    return np.around(validation_metrics['loss'] / max_iter, decimals=3)


def train(args):
    results_file = args.results_file
    #torch.manual_seed(args.random_seed) 
    
    # error metrics
    error_metrics = {'MSE': torch.nn.MSELoss(),
                    'MAE': torch.nn.L1Loss(),
                    'ME': MELoss()}

    if args.loss_key in ["GaussianNLL", "LaplacianNLL", "WeightedGaussianNLL", "WeightedLaplacianNLL"]:
        args.num_outputs = 2 # we predict mean and variance
        error_metrics['GaussianNLL'] = GaussianNLL()
        error_metrics['LaplacianNLL'] = LaplacianNLL()
    else:
        args.num_outputs = 1
    
    # particular case during fine-tuning
    if args.loss_key == "WeightedGaussianNLL":
        error_metrics['loss'] = WeightedGaussianNLL()
    elif args.loss_key == "WeightedLaplacianNLL":
        error_metrics['loss'] = WeightedLaplacianNLL()
    else:
        error_metrics['loss'] = error_metrics[args.loss_key]    
        
    # model   
    model = Net(args.arch, in_features=args.in_features, num_outputs=args.num_outputs, leaky_relu=args.leaky_relu, downsample=args.downsample)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
    best = 1e5 # for saving best model
    epoch_offset = 0
    
    
    # resume training from pth file
    if args.resume:
        checkpoint = torch.load(args.resume)  # load model to resume training
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_offset = checkpoint['epoch']
        loss = checkpoint['loss']
        #best = checkpoint['val_loss']
        model.to(device)
        with open(results_file, 'a') as f:
            f.write(f'Loading model {args.resume}, already trained for {epoch_offset} epochs \n')  
            
    wandb.init(settings=wandb.Settings(start_method="fork"), sync_tensorboard=True, project="biomass")  
    wandb.config.update(args)
    
    if args.freeze:
        print(f"freezing all but {args.freeze}")
        if args.arch in ["fcn_6_gaussian", "fcn_6_adf"]:
            model.model.conv_layers_mean.requires_grad_(False)
            model.model.conv_layers_var.requires_grad_(False)
            model.model.conv_output_var.requires_grad_(False)
        elif args.arch == "fcn_6":
            model.model.conv_layers.requires_grad_(False)
        else:
            raise NotImplementedError(f'freezing for {args.arch} was not implemented') 
    
    num_workers = multiprocessing.cpu_count()
    with open(results_file, 'a') as f:
        f.write(f'Number of available cores as known to the OS (virtual cores) is  {num_workers} \n')  
        f.write(f'Number of concrete (virtual) cores the thread (within the worker-process) is allowed to run: {len(os.sched_getaffinity(0))} \n')  
        
    writer = SummaryWriter(f"logs/{args.arch+args.addon}_{args.model_idx}_{args.loss_key}_{args.n_epochs+epoch_offset}_{args.learning_rate}_{args.batch_size}") 
        
    print('args:')
    print(args)
    print()
    
    train_dataset = GEDIDataset(args.dataset_path, type="train", args=args)
    val_dataset = GEDIDataset(args.dataset_path, type="val", args=args)
    
    args.mean_agbd = train_dataset.norm_values["mean_agbd"]
    args.sigma_agbd = train_dataset.norm_values["sigma_agbd"]
    
    with open(results_file, 'a') as f:
        f.write('len(train_dataset) = {}'.format(len(train_dataset)) + '\n')
        f.write('len(val_dataset) = {}'.format(len(val_dataset)) + '\n')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=min(32, num_workers//2))
    validation_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=min(32, num_workers//2))

    for epoch in trange(args.n_epochs):
        is_first = epoch == 0
        epoch += epoch_offset
        with open(results_file, 'a') as f:
            f.write('Starting epoch {}/{}'.format(epoch, args.n_epochs + epoch_offset) + '\n') 
            
        if is_first:
            loss = None
        else:
            loss = train_epoch(model, error_metrics, train_loader, optimizer, epoch, writer, args)
        val_loss = test_epoch(model, error_metrics, validation_loader, epoch, writer, args)
        scheduler.step()
        
        # save best model
        if val_loss < best:
            best = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'val_loss': val_loss}, f"pretrained_models/best_{args.saving_name}.pth")
        
        # backup model saving
        if epoch % 1 == 0:
            with open(results_file, 'a') as f:
                f.write(f'Saving backup model for epoch {epoch} \n')  
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'val_loss': val_loss}, 
                       f"pretrained_models/backup_{epoch}_{args.saving_name}.pth")
        
        # save last model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'loss': loss}, f"pretrained_models/last_{args.saving_name}.pth")

def main():
    parser = setup_parser()
    args, _ = parser.parse_known_args()

    if not os.path.exists('results/'):
        os.mkdir('results')
    
    if not os.path.exists('pretrained_models/'):
        os.mkdir('pretrained_models')
    
    addon = ""
    if args.arch=="unet":
        if args.leaky_relu:
            addon = "LeakyReLU"
        else:
            addon = "ReLU"
    if args.downsample:
        addon += f"_{args.downsample}pool"
    if not args.freeze:
        addon += f"_nofreeze"
    if args.tag:
        addon += ('_' + args.tag)
    args.addon = addon 
    
    # include latitude and longitude as input
    if args.latlon:
        args.in_features = 4
    elif args.lat:
        args.in_features = 2
    else:
        args.in_features = 1
    # include uncertainties from canopy height's predictions
    if args.include_std:
        args.in_features += 1
    
    if args.loss_key == "WeightedGaussianNLL":
        if args.sample_weighting_method == "ens":
            saving_name = f"{args.arch+args.addon}_{args.model_idx}_{args.loss_key}_{args.sample_weighting_method}_{args.beta}_{args.use_nb_of_classes}_{args.n_epochs}_{args.in_features}_{args.learning_rate}_{args.batch_size}"
        else:
            saving_name = f"{args.arch+args.addon}_{args.model_idx}_{args.loss_key}_{args.sample_weighting_method}_{args.use_nb_of_classes}_{args.n_epochs}_{args.in_features}_{args.learning_rate}_{args.batch_size}"
    else:
        saving_name = f"{args.arch+args.addon}_{args.model_idx}_{args.loss_key}_{args.n_epochs}_{args.in_features}_{args.learning_rate}_{args.batch_size}"
    if args.lat:
        saving_name += "_latOnly"
        
    args.results_file = f'results/results.{saving_name}_{args.results_name}.txt'
    args.saving_name = saving_name
    # Remove previous results
    for f in glob.glob(args.results_file):
        os.remove(f)
        
    train(args)


if __name__ == '__main__':
    main()
    
