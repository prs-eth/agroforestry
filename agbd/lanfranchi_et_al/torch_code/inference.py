import torch
import numpy as np
import os
import csv
import time
from osgeo import gdal
import rasterio
from skimage.transform import resize
from utils import gdal_process as gp
from utils.parser import setup_parser
from models.models import Net


class Inference:
    def __init__(self, net, weights, args):
        self.net = net
        self.weights = weights
        self.num_outputs = args.num_outputs
        self.nan_value = np.iinfo(np.uint16).max
        self.down_factor = args.down_factor
        self.tile_name = args.tile_name
        # get latitude and longitude from gdal repo
        # normalize input
        with open(os.path.join(args.dataset_path, 'normalization_values.csv'), mode='r') as infile:
            reader = csv.reader(infile)
            self.norm_values = {rows[0]: float(rows[1]) for rows in reader}
        
        # get canopy height predictions
        self.prediction_dir_tiles = "/cluster/work/igp_psr/nlang/S2_preds/GLOBAL_GEDI_latlon_True_models_5_FT_Lm_SRCB/GLOBAL_merge/preds_inv_var_mean/"
        refDataset_path = os.path.join(self.prediction_dir_tiles, '{}_pred.tif'.format(args.tile_name))
        ds = gdal.Open(refDataset_path)
        canopy_height = rasterio.open(refDataset_path).read(1)
        mask_no_data = canopy_height == 255
        self.mask_no_data = resize(mask_no_data,
            (ds.RasterYSize // self.down_factor, ds.RasterXSize // self.down_factor), order=0)
        
        #get std values (uncertainty)
        refDataset_path = os.path.join(self.prediction_dir_tiles, '{}_std.tif'.format(args.tile_name))
        uncertainty = rasterio.open(refDataset_path).read(1)  
        
        # normalization
        canopy_height = (canopy_height-self.norm_values["mean_can"])/self.norm_values["sigma_can"]         
        uncertainty = (uncertainty-self.norm_values["mean_std"])/self.norm_values["sigma_std"]
        
        if args.latlon:
            latitude, longitude = gp.create_latlon_mask(height=ds.RasterYSize, width=ds.RasterXSize, 
                                                        refDataset=ds)
            latitude = np.sin(np.pi*latitude / 180.0) # sin transformation is sufficient since latitude is not cyclic
            longitude_cos = np.cos(np.pi * longitude / 180.0)
            longitude_sin = np.sin(np.pi * longitude / 180.0)

            if args.include_std:
                img = np.stack([canopy_height, uncertainty, latitude, longitude_cos, longitude_sin], axis=0)
            else:
                img = np.stack([canopy_height, latitude, longitude_cos, longitude_sin], axis=0)
        elif args.lat:
            latitude, longitude = gp.create_latlon_mask(
                height=ds.RasterYSize, width=ds.RasterXSize, refDataset=ds)
            latitude = np.sin(np.pi * latitude / 180.0) # sin transformation is sufficient since latitude is not cyclic

            if args.include_std:
                img = np.stack([canopy_height, uncertainty, latitude], axis=0)
            else:
                img = np.stack([canopy_height, latitude], axis=0)
        else:
            img = np.expand_dims(canopy_height, axis=0)
            
        self.tile = img.astype(np.float32)
        self._init_network()

    def _init_network(self):
        print("Initializing weights from: {}...".format(self.weights))
        self.net.load_state_dict(torch.load(self.weights, map_location=lambda storage, loc: storage)['model_state_dict'])
        torch.cuda.set_device(device=0)
        self.net.cuda()

    def get_tile_prediction(self, patch_size=512, overlap=0.9):
        _, width, height = self.tile.shape
        down_factor = self.down_factor
        pred_width = width // down_factor
        pred_height = height // down_factor
        if patch_size % down_factor != 0:
            raise ValueError(f"patch dimension {patch_size} is not divisible by {down_factor}")
        predictions = np.zeros(shape=(width // down_factor, height // down_factor), dtype=np.float32)
        log_variances = np.zeros(shape=(width // down_factor, height // down_factor), dtype=np.float32)
        # Iterate over tile
        # steps = int(width//(patch_size*overlap))
        prediction_size = (patch_size - int(patch_size*overlap)) // (2 * down_factor)
        prediction_patch_size = patch_size // down_factor
        print(f"prediction_size = {prediction_size}, prediction_patch_size = {prediction_patch_size}")
        self.net.eval()
        # with tqdm(total=steps*steps) as bar:
        # Top left tile
        patch = self.tile[:, :patch_size, :patch_size]
        patch = torch.tensor(patch[None, ...]).cuda()
        with torch.no_grad():
            outputs = self.net(patch).squeeze().cpu().numpy()
        if self.num_outputs == 2:
            predictions[:prediction_patch_size, :prediction_patch_size] = outputs[0, :, :]
            log_variances[:prediction_patch_size, :prediction_patch_size] = outputs[1, :, :]
        else:
            predictions[:prediction_patch_size, :prediction_patch_size] = outputs   
        # bar.update(1)
        # First row
        for x in range(int(patch_size * overlap), width - patch_size, int(patch_size * overlap)):
            patch = self.tile[:, :patch_size, x:x+patch_size]
            patch = torch.tensor(patch[None, ...]).cuda()
            x = x//self.down_factor
            with torch.no_grad():
                outputs = self.net(patch).squeeze().cpu().numpy()
            if self.num_outputs == 2:
                predictions[:prediction_patch_size, x + prediction_size:x + prediction_patch_size] = outputs[0, :, prediction_size:]
                log_variances[:prediction_patch_size, x + prediction_size:x + prediction_patch_size] = outputs[1, :, prediction_size:]
            else:
                predictions[:prediction_patch_size, x + prediction_size:x + prediction_patch_size] = outputs[:, prediction_size:]
            # bar.update(1)
        # First row last tile
        patch = self.tile[:, :patch_size, -patch_size:]
        patch = torch.tensor(patch[None, ...]).cuda()
        with torch.no_grad():
            outputs = self.net(patch).squeeze().cpu().numpy()
        if self.num_outputs == 2:
            predictions[:prediction_patch_size, -prediction_patch_size+prediction_size:] = outputs[0, :, prediction_size:]
            log_variances[:prediction_patch_size, -prediction_patch_size+prediction_size:] = outputs[1, :, prediction_size:]
        else:
            predictions[:prediction_patch_size, -prediction_patch_size+prediction_size:] = outputs[:, prediction_size:]
        # bar.update(1)
        # Loop over row 1 to last-1 row
        for y in range(int(patch_size*overlap), height-patch_size, int(patch_size*overlap)):
            # First tile left
            patch = self.tile[:, y:y+patch_size, 0:patch_size]
            patch = torch.tensor(patch[None, ...]).cuda()
            y_pred = y//self.down_factor
            with torch.no_grad():
                outputs = self.net(patch).squeeze().cpu().numpy()
            if self.num_outputs == 2:
                predictions[y_pred:y_pred+prediction_patch_size, 0:prediction_patch_size] = outputs[0, :, :]
                log_variances[y_pred:y_pred+prediction_patch_size, 0:prediction_patch_size] = outputs[1, :, :]
            else:
                predictions[y_pred:y_pred+prediction_patch_size, 0:prediction_patch_size] = outputs
            # bar.update(1)
            for x in range(int(patch_size * overlap), width-patch_size, int(patch_size * overlap)):
                patch = self.tile[:, y:y + patch_size, x:x+patch_size]
                patch = torch.tensor(patch[None, ...]).cuda()
                x = x//self.down_factor
                with torch.no_grad():
                    outputs = self.net(patch).squeeze().cpu().numpy()
                if self.num_outputs == 2:
                    predictions[y_pred+prediction_size:y_pred + prediction_patch_size, x+prediction_size:x+prediction_patch_size] = outputs[0, prediction_size:, prediction_size:]
                    log_variances[y_pred+prediction_size:y_pred + prediction_patch_size, x+prediction_size:x+prediction_patch_size] = outputs[1, prediction_size:, prediction_size:]
                else:
                    predictions[y_pred+prediction_size:y_pred + prediction_patch_size, x+prediction_size:x+prediction_patch_size] = outputs[prediction_size:, prediction_size:]
                # bar.update(1)
            # Last tile right
            patch = self.tile[:, y:y + patch_size, width-patch_size:width]
            patch = torch.tensor(patch[None, ...]).cuda()
            with torch.no_grad():
                outputs = self.net(patch).squeeze().cpu().numpy()
            if self.num_outputs == 2:
                predictions[y_pred+prediction_size:y_pred + prediction_patch_size, pred_width-prediction_patch_size+prediction_size:pred_width] = outputs[0, prediction_size:, prediction_size:]
                log_variances[y_pred+prediction_size:y_pred + prediction_patch_size, pred_width-prediction_patch_size+prediction_size:pred_width] = outputs[1, prediction_size:, prediction_size:]
            else:
                predictions[y_pred+prediction_size:y_pred + prediction_patch_size, pred_width-prediction_patch_size+prediction_size:pred_width] = outputs[prediction_size:, prediction_size:]
            # bar.update(1)
        # Left bottom tile
        patch = self.tile[:, -patch_size:, 0:patch_size]
        patch = torch.tensor(patch[None, ...]).cuda()
        with torch.no_grad():
            outputs = self.net(patch).squeeze().cpu().numpy()
        if self.num_outputs == 2:
            predictions[-prediction_patch_size+prediction_size:, 0:prediction_patch_size] = outputs[0, prediction_size:, :]
            log_variances[-prediction_patch_size+prediction_size:, 0:prediction_patch_size] = outputs[1, prediction_size:, :]
        else:
            predictions[-prediction_patch_size+prediction_size:, 0:prediction_patch_size] = outputs[prediction_size:, :]
        # bar.update(1)
        # Last row
        for x in range(int(patch_size * overlap), width - patch_size, int(patch_size * overlap)):
            patch = self.tile[:, -patch_size:, x:x + patch_size]
            patch = torch.tensor(patch[None, ...]).cuda()
            x = x//self.down_factor
            with torch.no_grad():
                outputs = self.net(patch).squeeze().cpu().numpy()
            if self.num_outputs == 2:
                predictions[-prediction_patch_size+prediction_size:, x + prediction_size:x + prediction_patch_size] = outputs[0, prediction_size:, prediction_size:]
                log_variances[-prediction_patch_size+prediction_size:, x + prediction_size:x + prediction_patch_size] = outputs[1, prediction_size:, prediction_size:]
            else:
                predictions[-prediction_patch_size+prediction_size:, x + prediction_size:x + prediction_patch_size] = outputs[prediction_size:, prediction_size:]
            # bar.update(1)
        # Bottom right tile
        patch = self.tile[:, height-patch_size:height, width - patch_size:width]
        patch = torch.tensor(patch[None, ...]).cuda()
        with torch.no_grad():
            outputs = self.net(patch).squeeze().cpu().numpy()
        if self.num_outputs == 2:
            predictions[pred_height - prediction_patch_size+prediction_size:pred_height, pred_width - prediction_patch_size+prediction_size:pred_width] = outputs[0, prediction_size:, prediction_size:]
            log_variances[pred_height - prediction_patch_size+prediction_size:pred_height, pred_width - prediction_patch_size+prediction_size:pred_width] = outputs[1, prediction_size:, prediction_size:]
        else:
            predictions[pred_height - prediction_patch_size+prediction_size:pred_height, pred_width - prediction_patch_size+prediction_size:pred_width] = outputs[prediction_size:, prediction_size:]
        # bar.update(1)

        # Denormalize
        predictions = np.clip(predictions*self.norm_values["sigma_agbd"] + self.norm_values["mean_agbd"], 0, None) # agbd predictions need to be >=0
        var = np.exp(log_variances)*self.norm_values["sigma_agbd"]**2
        
        # Mask no data
        predictions[self.mask_no_data] = self.nan_value
        var[self.mask_no_data] = self.nan_value

        return predictions, var
    
    def get_esa_mask(self):
        esa_landcover_path = "/cluster/work/igp_psr/data/ESA_WorldCover/data/sentinel2_tiles"
        # 105 tiles in the Pacific Ocean are missing and are thus to be found in following directory (different from above due to writing permision not enabled)
        esa_landcover_path2 = "/cluster/work/igp_psr/clanfranchi/gdal_processing/sentinel2_tiles"
        try:
            # Get ESA mask for water, ice and built-up areas
            if os.path.exists(os.path.join(esa_landcover_path, f"ESA_WorldCover_10m_2020_v100_{self.tile_name}.tif")):
                esa_landcover = rasterio.open(os.path.join(esa_landcover_path, f"ESA_WorldCover_10m_2020_v100_{self.tile_name}.tif")).read(1)
                ds = gdal.Open(os.path.join(esa_landcover_path, f"ESA_WorldCover_10m_2020_v100_{self.tile_name}.tif"))
            else:
                esa_landcover = rasterio.open(os.path.join(esa_landcover_path2, f"ESA_WorldCover_10m_2020_v100_{self.tile_name}.tif")).read(1)
                ds = gdal.Open(os.path.join(esa_landcover_path2, f"ESA_WorldCover_10m_2020_v100_{self.tile_name}.tif"))
            nodata_mask = esa_landcover == 0
            water_mask = esa_landcover == 80
            ice_mask = esa_landcover == 70
            builtup_mask = esa_landcover == 50
            esa_mask = np.logical_or.reduce((nodata_mask, water_mask, ice_mask, builtup_mask))
        except:
            print(f"Failed tile {self.tile_name}")
            
        esa_mask = resize(esa_mask,(ds.RasterYSize//self.down_factor,ds.RasterXSize//self.down_factor),order=0)
        return esa_mask
        
    
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
    if args.downsample:
        addon += f"_{args.downsample}pool"
        args.down_factor = 5
    else:
        args.down_factor = 1
    args.addon = addon 
    
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
    if args.loss_key in ["GaussianNLL", "LaplacianNLL", "WeightedGaussianNLL"]:
        args.num_outputs = 2 # we predict mean and variance
    else:
        args.num_outputs = 1
    
    print("Predicting for tile ", args.tile_name)
    model = Net(args.arch, in_features=args.in_features, num_outputs=args.num_outputs, leaky_relu=args.leaky_relu, downsample=args.downsample)
    
    # one model only
    if args.n_models == 1:
        model_path = os.path.join(args.model_path, args.model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'The pth file {model_path} does not exist.')
        inference = Inference(model, model_path, args)
        start = time.time()
        patch_size = 512
        preds, var = inference.get_tile_prediction(patch_size)
        print(f"Prediction done in {int(time.time()-start)}s")
        
        saving_path = f"{args.output_path}/{args.arch+args.addon}_{args.loss_key}_{args.in_features}"
    
    # ensemble model 
    else:
        preds_agbd = []
        preds_var = []
        for model_idx in range(1, args.n_models+1):
            if args.loss_key == "WeightedGaussianNLL":
                if args.sample_weighting_method == "ens":
                    model_name = f"last_{args.arch+args.addon}_{model_idx}_{args.loss_key}_{args.sample_weighting_method}_{args.beta}_True_{args.n_epochs}_{args.in_features}_{args.learning_rate}_{args.batch_size}"
                else:
                    model_name = f"last_{args.arch+args.addon}_{model_idx}_{args.loss_key}_{args.sample_weighting_method}_True_{args.n_epochs}_{args.in_features}_{args.learning_rate}_{args.batch_size}"
            else:
                model_name = f"best_{args.arch+args.addon}_{model_idx}_{args.loss_key}_{args.n_epochs}_{args.in_features}_{args.learning_rate}_{args.batch_size}"
            if args.lat:
                model_name += "_latOnly"
            model_name += ".pth"
            model_path = os.path.join(args.model_path, model_name)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'The pth file {model_path} does not exist.')
            inference = Inference(model, model_path, args)
            start = time.time()
            patch_size = 510
            overlap = 0.9
            agbd, var = inference.get_tile_prediction(patch_size, overlap)
            print(f"Prediction done in {int(time.time()-start)}s")
            preds_agbd.append(agbd)
            preds_var.append(var)
            
        preds_agbd = np.array(preds_agbd, dtype=np.float32)
        preds_var = np.array(preds_var, dtype=np.float32)
        
        preds = np.mean(preds_agbd, axis=0) # ensemble prediction
        epistemic_var = np.var(preds_agbd, axis=0)
        aleatoric_var = np.mean(preds_var, axis=0)
        var = epistemic_var + aleatoric_var # predictive variance
        
        if args.loss_key == "WeightedGaussianNLL":
            if args.sample_weighting_method == "ens":
                saving_path = f"{args.output_path}/{args.arch+args.addon}_ensemble_{args.loss_key}_{args.sample_weighting_method}_{args.beta}_{args.in_features}"
            else:
                saving_path = f"{args.output_path}/{args.arch+args.addon}_ensemble_{args.loss_key}_{args.sample_weighting_method}_{args.in_features}"
        else:
            saving_path = f"{args.output_path}/{args.arch+args.addon}_ensemble_{args.loss_key}_{args.in_features}"
    
    
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
        
    esa_mask = inference.get_esa_mask()
    preds[esa_mask] = inference.nan_value
    
    # save AGBD predictions to Geotiff format
    canopy_tile_path = os.path.join(inference.prediction_dir_tiles, f"{args.tile_name}_pred.tif")
    with rasterio.open(canopy_tile_path) as f:
        meta = f.meta
        # scale image transform
        transform = f.transform * f.transform.scale(
            (f.width / preds.shape[-1]),
            (f.height / preds.shape[-2])
        )

        # update meta information
        meta['height'] = preds.shape[0]
        meta['width'] = preds.shape[1]
        meta['transform'] = transform
        
    meta.update(driver='GTiff', dtype=np.uint16, count=1, compress='lzw')
    stacked_img = os.path.join(saving_path, f"{args.tile_name}_agbd.tif")
    with rasterio.open(stacked_img, 'w', **meta) as f:
        f.write(preds.astype(np.uint16), 1)
        f.nodata = inference.nan_value
    
    mask_no_data = preds == inference.nan_value
    std = np.sqrt(var)
    std[mask_no_data] = inference.nan_value
    # save std predictions if Gaussian loss
    if args.num_outputs==2:
        stacked_img = os.path.join(saving_path, f"{args.tile_name}_std.tif")
        with rasterio.open(stacked_img, 'w', **meta) as f:
            f.write(std.astype(np.uint16), 1)
            f.nodata = inference.nan_value
        
        if args.save_variances_type:
            epistemic_std = np.sqrt(epistemic_var)
            epistemic_std[mask_no_data] = inference.nan_value
            aleatoric_std = np.sqrt(aleatoric_var)
            aleatoric_std[mask_no_data] = inference.nan_value
            stacked_img = os.path.join(saving_path, f"{args.tile_name}_epistemic_std.tif")
            with rasterio.open(stacked_img, 'w', **meta) as f:
                f.write(epistemic_std.astype(np.uint16), 1)
                f.nodata = inference.nan_value
            stacked_img = os.path.join(saving_path, f"{args.tile_name}_aleatoric_std.tif")
            with rasterio.open(stacked_img, 'w', **meta) as f:
                f.write(aleatoric_std.astype(np.uint16), 1)
                f.nodata = inference.nan_value


if __name__ == '__main__':
    main()