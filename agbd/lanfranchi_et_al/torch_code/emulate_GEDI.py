import torch
import numpy as np
import collections
import os
import time
import glob
import multiprocessing
from utils.parser import setup_parser
from utils.loss import MELoss, GaussianNLL, LaplacianNLL
from utils.transforms import denormalize
from models.models import Net
from dataset import GEDIDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

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
    for i, (images, labels, _) in enumerate(train_loader):
        images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
        predictions = model(images).squeeze()
        agbd, se = predictions[:,0], predictions[:,1]
        gt, gt_se = labels[:,0], labels[:,1] # GEDI agbd (mean) and agbd standard error (se) reference
        mask = torch.isnan(gt)
        
        loss1 = error_metrics['loss'](agbd[~mask], gt[~mask])
        loss2 = error_metrics['loss'](se[~mask], gt_se[~mask])
        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i%1000==0:
            with open(args.results_file, 'a') as f:
                f.write(f'Loss for batch {i}/{total_step} of epoch {epoch} is {np.round(loss1.item(), 3)} + {np.round(loss2.item(), 3)} = {np.round(loss.item(), 3)} | Time is: {np.round(time.time() - start,2)}s \n')
        
        # denormalized labels and predictions
        agbd_ = torch.clamp(denormalize(agbd, args.mean_agbd, args.sigma_agbd), 0)  # agbd is >=0
        gt_ = denormalize(gt, args.mean_agbd, args.sigma_agbd)
        se_ = torch.clamp(denormalize(se, args.mean_agbd_se, args.sigma_agbd_se), 0) # standard error is >=0
        gt_se_ = denormalize(gt_se, args.mean_agbd_se, args.sigma_agbd_se)
        
        # compute metrics on every batch and add to running sum
        for metric in error_metrics:
            training_metrics[metric+"_AGBD"] += error_metrics[metric](agbd_[~mask], gt_[~mask]).item()
            training_metrics[metric+"_SE"] += error_metrics[metric](se_[~mask], gt_se_[~mask]).item()
                
        # test
        if i == max_iter:
            break
        
    for metric in training_metrics.keys():
        writer.add_scalar(f'training/{metric}', training_metrics[metric] / max_iter, global_step=epoch) 
    
    with open(args.results_file, 'a') as f:
        f.write(f'gt shape: {gt.shape}, gt_se shape: {gt_se.shape} \n')

    return np.around(running_loss / max_iter, decimals=3)

def test_epoch(model, error_metrics, test_loader, epoch, writer, args):
    # init running error
    validation_metrics = collections.defaultdict(lambda: 0)
    
    model.eval()
    max_iter = int(args.n_iter*0.1) if args.n_iter is not None else len(test_loader)
    with torch.no_grad():
        for i, (images, labels, _) in enumerate(test_loader):
            images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
            predictions = model(images).squeeze()
            agbd, se = predictions[:,0], predictions[:,1]
            gt, gt_se = labels[:,0], labels[:,1] # GEDI agbd (mean) and agbd standard error (se) reference
            mask = torch.isnan(gt)
            
            # denormalized labels and predictions
            agbd_ = torch.clamp(denormalize(agbd, args.mean_agbd, args.sigma_agbd), 0) # agbd is >=0
            gt_ = denormalize(gt, args.mean_agbd, args.sigma_agbd)
            se_ = torch.clamp(denormalize(se, args.mean_agbd_se, args.sigma_agbd_se), 0) # standard error is >=0
            gt_se_ = denormalize(gt_se, args.mean_agbd_se, args.sigma_agbd_se)
            
            for metric in error_metrics:
                validation_metrics[metric+"_AGBD"] += error_metrics[metric](agbd_[~mask], gt_[~mask]).item()
                validation_metrics[metric+"_SE"] += error_metrics[metric](se_[~mask], gt_se_[~mask]).item()

            if i == max_iter:
                break

        for metric in validation_metrics.keys():
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
    
    error_metrics['loss'] = error_metrics[args.loss_key]
        
    model = Net(args.arch, in_features=args.in_features, num_outputs=2, leaky_relu=args.leaky_relu)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best = 1e5 # for saving best model
    epoch_offset = 0
    
    # resume training from pth file
    if len(args.resume):
        checkpoint = torch.load(args.resume)  # load model to resume training
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_offset = checkpoint['epoch']
        loss = checkpoint['loss']
        best = checkpoint['val_loss']
        model.to(device)
        with open(results_file, 'a') as f:
            f.write(f'Loading model {args.resume}, already trained for {epoch_offset} epochs \n')  
        wandb.init(settings=wandb.Settings(start_method="fork"), sync_tensorboard=True, project="EmulateGEDIL4B", entity="clanfranchi", id=args.run_id, resume="must")  
        wandb.config.update(args, allow_val_change=True)
    
    else:
        wandb.init(settings=wandb.Settings(start_method="fork"), sync_tensorboard=True, project="EmulateGEDIL4B", entity="clanfranchi")  
        wandb.config.update(args)
    
    num_workers = multiprocessing.cpu_count()
    with open(results_file, 'a') as f:
        f.write(f'Number of available cores as known to the OS (virtual cores) is  {num_workers} \n')  
        f.write(f'Number of concrete (virtual) cores the thread (within the worker-process) is allowed to run: {len(os.sched_getaffinity(0))} \n')  
        
    writer = SummaryWriter(f"logs/{args.arch+args.addon}_{args.model_idx}_{args.loss_key}_{args.n_epochs+epoch_offset}_{args.in_features}_{args.learning_rate}_{args.batch_size}") 
        
    train_dataset = GEDIDataset(args.dataset_path, type="train", latlon=args.latlon, include_std=args.include_std, predict_agbd_se=True, 
                                normalize_input=args.normalize_input, normalize_gt=args.normalize_gt)
    val_dataset = GEDIDataset(args.dataset_path, type="val", latlon=args.latlon, include_std=args.include_std, predict_agbd_se=True, 
                              normalize_input=args.normalize_input, normalize_gt=args.normalize_gt)
    
    args.mean_agbd = train_dataset.norm_values["mean_agbd"]
    args.sigma_agbd = train_dataset.norm_values["sigma_agbd"]
    args.mean_agbd_se = train_dataset.norm_values["mean_agbd_se"]
    args.sigma_agbd_se = train_dataset.norm_values["sigma_agbd_se"]
    
    with open(results_file, 'a') as f:
        f.write('len(train_dataset) = {}'.format(len(train_dataset)) + '\n')
        f.write('len(val_dataset) = {}'.format(len(val_dataset)) + '\n')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=min(32, num_workers//2))
    validation_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=min(32, num_workers//2))

    for epoch in range(args.n_epochs):
        epoch += epoch_offset
        with open(results_file, 'a') as f:
            f.write('Starting epoch {}/{}'.format(epoch, args.n_epochs + epoch_offset) + '\n') 
            
        loss = train_epoch(model, error_metrics, train_loader, optimizer, epoch, writer, args)
        val_loss = test_epoch(model, error_metrics, validation_loader, epoch, writer, args)
        
        # save best model
        if val_loss < best:
            best = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'val_loss': val_loss}, f"pretrained_models/best_{args.arch+args.addon}_{args.model_idx}_{args.loss_key}_{args.n_epochs}_{args.in_features}_{args.learning_rate}_{args.batch_size}.pth")
        
        # backup model saving
        if epoch % 100 == 0:
            with open(results_file, 'a') as f:
                f.write(f'Saving backup model for epoch {epoch} \n')  
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'val_loss': val_loss}, 
                       f"pretrained_models/backup_{args.arch+args.addon}_{args.model_idx}_{args.loss_key}_{epoch}_{args.in_features}_{args.learning_rate}_{args.batch_size}.pth")
        
        # save last model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'loss': loss}, f"pretrained_models/last_{args.arch+args.addon}_{args.model_idx}_{args.loss_key}_{args.n_epochs}_{args.in_features}_{args.learning_rate}_{args.batch_size}.pth")

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
    
    addon += "_emulateGEDI"
        
    args.addon = addon 
    
    # model
    if args.latlon:
        args.in_features = 4
    else:
        args.in_features = 1
    
    if args.include_std:
        args.in_features +=1
    
    args.results_file = f'results/results.{args.arch+args.addon}_{args.model_idx}_{args.loss_key}_{args.n_epochs}_{args.in_features}_{args.learning_rate}_{args.batch_size}_{args.n_iter}_{args.results_name}.txt'
    # Remove previous results
    for f in glob.glob(args.results_file):
        os.remove(f)
        
    train(args)

if __name__ == '__main__':
    main()
    