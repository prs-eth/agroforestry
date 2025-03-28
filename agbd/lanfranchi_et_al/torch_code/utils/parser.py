import argparse

def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", default='./tmp/', help="output directory for the experiment")
    parser.add_argument("--tag", default='')
    parser.add_argument("--wd", default=0.0, type=float)

    # dataset file
    parser.add_argument("--tiles_indices_path", default="", help="path to pickle file with cumulative number of images for each tile")
    parser.add_argument("--dataset_path", default="", help="path to the folder with merged h5 files for each key (agbd, canopy_height, etc.)")
    parser.add_argument("--normalize_input", type=str2bool, nargs='?', const=True, default=True, help="True: normalize canopy height")
    parser.add_argument("--normalize_gt", type=str2bool, nargs='?', const=True, default=True, help="True: normalize agbd (ground truth)")
    parser.add_argument("--latlon", type=str2bool, nargs='?', const=True, default=True, help="True: include lat/lon data as features")
    parser.add_argument("--lat", type=str2bool, nargs='?', const=True, default=True, help="True: include latitude data as feature only (no longitude)")
    parser.add_argument("--include_std", type=str2bool, nargs='?', const=True, default=True, help="True: include std from canopy height preds as feature")
    parser.add_argument("--predict_agbd_se", type=str2bool, nargs='?', const=True, default=False, help="True: predict GEDI's standard error as additional output")
    
    # results
    parser.add_argument("--results_name", default="", help="additional name of the results file")
    parser.add_argument("--run_id", default="", help="wandb id for resuming training")
    
    # model
    parser.add_argument("--arch", default='fcn_6', type=str, choices=['fcn_4', 'fcn_6', 'fcn_6_adf', 'fcn_6_gaussian', 'fcn_6_kernel', 'unet'],
                        help='Network architecture. We have fcn (Fully Convolutional Network) and unet.')
    parser.add_argument("--downsample", default="", type=str, choices=["", "max", "average"], 
                        help="Whether to add a Pooling layer (max or avg) before final convolutional layer")
    parser.add_argument("--leaky_relu", type=str2bool, nargs='?', const=True, default=False, 
                        help="True: use Leaky ReLU activation for UNet model")
    parser.add_argument("--resume", default="", type=str, help='path to the saved model to resume training from')
    
    # training
    parser.add_argument("--n_epochs", default=100, type=int, help="number of epochs")
    parser.add_argument("--n_iter", default=None, type=int, help="number of iterations per epoch")
    parser.add_argument("--batch_size", default=256, type=int, help="size of batch")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--loss_key", default='MSE', type=str) 
    parser.add_argument("--model_idx", default=1, type=int, help='Numerotation of models for ensemble')
    #parser.add_argument("--num_workers", default=8, help="Number of workers for pytorch Dataloader")
    
    # fine-tuning
    parser.add_argument("--sample_weighting_method", default='ifns', type=str, choices=['ins', 'isns', 'ifns', 'ens'], 
                        help='method to get sample weights')
    parser.add_argument("--beta", default=0.99, type=float, help='beta value for ens method')
    parser.add_argument("--use_nb_of_classes", type=str2bool, nargs='?', const=True, default=True, 
                        help="whether to multiply by nb of classes for normalization")
    parser.add_argument("--freeze", default="", type=str, choices=["", "last_layer"], help="select which layers to fine-tune")
    
    # evaluate
    parser.add_argument('--model_path', default="", type=str, help="path to the folder where models are saved")
    parser.add_argument('--model_name', default="", type=str, help="name of the saved pth file of a trained model")
    parser.add_argument('--model_nb', default=0, type=int, help="index of the model for ensemble evaluation")
    parser.add_argument('--aggregate', type=str2bool, nargs='?', const=True, default=True, 
                        help="True: aggregate predictions of the 9x9 pixels around GEDI's ground truth")
    
    # predict
    parser.add_argument("--tile_name", type=str, default="", help="path to tile to predict")
    parser.add_argument("--clip", type=str2bool, nargs='?', const=True, default=True, help="True: clip negative AGBD to 0")
    parser.add_argument("--output_path", default="", help="path to the folder where to save the predictions")
    parser.add_argument("--save_variances_type", type=str2bool, nargs='?', const=True, default=True, help="True: save epistemic and aleatoric variances in difference files")
    # predict ensemble
    parser.add_argument("--n_models", type=int, default=10, help="number of models to ensemble")
    
    
    parser.add_argument("--random_seed", default=42, type=int, help="for train-val split")
    
    return parser


# --- Helper functions to parse arguments ---

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
