import torch
import numpy as np
import h5py
import tables as tb
import os
import time
import glob
import multiprocessing
from utils.parser import setup_parser
from utils.transforms import denormalize
from models.models import Net
from dataset import GEDIDataset
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
FILTERS = tb.Filters(complib='zlib', complevel=4)

"""
This code will generate predictions for test dataset that is in the
same format as for train and validation (i.e., a big h5 file for each 
field and 15x15 patches centered at each ground truth's location)
"""

def evaluate(model, test_loader, args):
    model.eval()
    
    max_iter = args.n_iter if args.n_iter is not None else len(test_loader)
    eval_predictions = None
    start = time.time()
    
    saving_path = "data/predictions"
    if not os.path.exists(saving_path):
        # Create a new directory because it does not exist 
        os.makedirs(saving_path)
    
    with tb.File(os.path.join(saving_path, f"preds_{args.saving_name}.h5"),mode='w') as h5fw:
        with torch.no_grad():
            for i, (images, labels, _) in enumerate(test_loader):
                images, labels = images.to(device, dtype=torch.float32).squeeze(), labels.to(device, dtype=torch.float32).squeeze()
                predictions = model(images).squeeze()
                if args.downsample:
                    M, N = labels.shape[1:]
                    K = 3
                    MK = M//K
                    NK = N//K
                    labels = torch.nanmean(labels.reshape(labels.shape[0], MK, K, NK, K), axis=(1, 3))
            
                if args.num_outputs==2:
                    predictions, log_variances = predictions[:,0], predictions[:,1]
                
                # denormalize labels and predictions
                predictions = denormalize(predictions, args.mean_agbd, args.sigma_agbd)
                labels = denormalize(labels, args.mean_agbd, args.sigma_agbd)
                predictions = torch.clamp(predictions, 0) # preds should be >=0
                
                n = predictions.shape[1]
                # aggregate values on a 9x9 pixel square (~100m precision)
                if args.aggregate:      
                    predictive_var = np.var(predictions.cpu().numpy()[:, n//2-4:n//2+5, n//2-4:n//2+5], axis=(1,2))          
                    predictions = np.mean(predictions.cpu().numpy()[:, n//2-4:n//2+5, n//2-4:n//2+5], axis=(1,2))
                    labels = np.nanmean(labels.cpu().numpy()[:, n//2-4:n//2+5, n//2-4:n//2+5], axis=(1,2))
                else:
                    predictions = predictions.cpu().numpy()[:, n//2, n//2]
                    labels = labels.cpu().numpy()[:, n//2, n//2]
                
                # dernormalize variances
                if args.num_outputs==2:
                    var = torch.exp(log_variances)*args.sigma_agbd**2
                    if args.aggregate:     
                        aleatoric_var = np.mean(var.cpu().numpy()[:, n//2-4:n//2+5, n//2-4:n//2+5], axis=(1,2))
                        var = aleatoric_var + predictive_var
                    else:
                        var = var.cpu().numpy()[:, n//2, n//2]

                if eval_predictions is None:
                    with open(args.results_file, 'a') as f:
                        f.write(f'Size of preds is {predictions.shape}, size of gt is {labels.shape} \n')
                    preds = h5fw.create_earray(h5fw.root,"preds", 
                                                obj=predictions, filters=FILTERS)
                    gt = h5fw.create_earray(h5fw.root,"gt",  
                                                obj=labels, filters=FILTERS)
                    if args.num_outputs==2:
                        vars = h5fw.create_earray(h5fw.root,"vars", 
                                                    obj=var, filters=FILTERS)
                    
                    eval_predictions = 1
                else:
                    preds.append(predictions)
                    gt.append(labels)
                    if args.num_outputs==2:
                        vars.append(var)
                
                if i == max_iter:
                    break
                
                if i%1000==0:
                    with open(args.results_file, 'a') as f:
                        f.write(f'Batch {i}/{len(test_loader)} | Time is: {np.round(time.time() - start,2)}s \n')
                

def main():
    parser = setup_parser()
    args, _ = parser.parse_known_args()
    
    # type of ReLU function for UNet architecture
    addon = ""
    if args.arch=="unet":
        if args.leaky_relu:
            addon = "LeakyReLU"
        else:
            addon = "ReLU"
    addon += args.results_name
    
    if args.downsample:
        addon += f"_{args.downsample}pool"
        
    args.addon = addon 
    
    if args.model_nb:
        args.addon += f"_{args.model_nb}"
    
    # number of input features
    if args.latlon:
        args.in_features = 4
    elif args.lat:
        args.in_features = 2
    else:
        args.in_features = 1
    # include uncertainties from canopy height's predictions
    if args.include_std:
        args.in_features +=1
    
    # loss
    if args.loss_key in ["GaussianNLL", "WeightedGaussianNLL", "LaplacianNLL"]:
        args.num_outputs = 2 # we predict mean and variance
    else:
        args.num_outputs = 1
    
    if args.loss_key == "WeightedGaussianNLL":
        if args.sample_weighting_method == "ens":
            saving_name = f"{args.arch+args.addon}_{args.loss_key}_{args.sample_weighting_method}_{args.beta}_{args.use_nb_of_classes}_{args.aggregate}_{args.in_features}"
        else:
            saving_name = f"{args.arch+args.addon}_{args.loss_key}_{args.sample_weighting_method}_{args.use_nb_of_classes}_{args.aggregate}_{args.in_features}"
    else:
        saving_name = f"{args.arch+args.addon}_{args.loss_key}_{args.aggregate}_{args.in_features}"
    if args.lat:
        saving_name += "_latOnly"
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'The pth file {args.model_path} does not exist.')
        
    args.saving_name = saving_name
    args.results_file = f'results/eval.{saving_name}.txt'
    # Remove previous results
    for f in glob.glob(args.results_file):
        os.remove(f)
    
    model = Net(args.arch, in_features=args.in_features, num_outputs=args.num_outputs, leaky_relu=args.leaky_relu, downsample=args.downsample)
    model.to(device)
    checkpoint = torch.load(args.model_path)  # load model used to predict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    num_workers = multiprocessing.cpu_count()
    test_dataset = GEDIDataset(args.dataset_path, type="test", args=args)
    args.sigma_agbd = test_dataset.norm_values["sigma_agbd"]
    args.mean_agbd = test_dataset.norm_values["mean_agbd"]
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=max(8, num_workers//2))
    
    start = time.time()
    evaluate(model, test_loader, args)
    print(f'Evaluation done in {np.round((time.time()-start)/60, 2)}min')


if __name__ == '__main__':
    main()