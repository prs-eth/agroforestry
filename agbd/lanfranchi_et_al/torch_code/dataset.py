from torch.utils.data import Dataset
import numpy as np
import h5py
import os
import csv
from utils.sample_weighting import get_weights_for_sample
import yaml
import json


class GEDIDataset(Dataset):
    def __init__(self, dataset_path, type, args):
        if type not in ["train", "val", "test"]:
            raise ValueError('The type {} is not implemented.'.format(type))
        self.dataset_path = dataset_path
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError('The folder {} does not exist.'.format(self.dataset_path))
        
        self.type = type
        self.latlon = args.latlon
        self.lat = args.lat
        self.include_std = args.include_std
        self.predict_agbd_se = args.predict_agbd_se
        
        with open(os.path.join(self.dataset_path,'sample_weights.json')) as fh:
            obj = json.load(fh)
            self.bins = np.array(obj['bins']).astype('float32')
            self.weights = np.array(obj['weights']).astype('float32')
            self.weights = np.pad(self.weights, (0,1), 'symmetric')
        
        print(self.weights)
        print(self.bins)
        
        # for my dataset
        if not os.path.exists(os.path.join(self.dataset_path,'normalization_values.csv')):
            raise FileNotFoundError('The file {} does not exist.'.format("normalization_values.csv"))
            
        with open(os.path.join(self.dataset_path,'normalization_values.csv'), mode='r') as infile:
            reader = csv.reader(infile)
            self.norm_values = {rows[0]:float(rows[1]) for rows in reader}
            
        if not args.normalize_input:
            self.norm_values["mean_can"]=0.0
            self.norm_values["sigma_can"]=1.0
            
        if not args.normalize_gt:
            self.norm_values["mean_agbd"]=0.0
            self.norm_values["sigma_agbd"]=1.0
            self.norm_values["mean_agbd_se"]=0.0
            self.norm_values["sigma_agbd_se"]=1.0
        
        agbd = h5py.File(os.path.join(self.dataset_path,f'agbd_{self.type}.h5'),'r')
        self.total = agbd.get("agbd").shape[0]
        agbd.close()
        
    def __len__(self):
        return self.total
    
    def __getitem__(self, idx):
        # c.f. https://github.com/pytorch/pytorch/issues/11929
        # "the actual file opening has to happen inside of the__getitem__function of the Dataset wrapper"   
        agbd = h5py.File(os.path.join(self.dataset_path,f'agbd_{self.type}.h5'),'r')
        agbd_se = h5py.File(os.path.join(self.dataset_path,f'agbd_se_{self.type}.h5'),'r')
        lat = h5py.File(os.path.join(self.dataset_path,f'lat_{self.type}.h5'),'r')
        lon = h5py.File(os.path.join(self.dataset_path,f'lon_{self.type}.h5'),'r')
        canopy_height = h5py.File(os.path.join(self.dataset_path,f'canopy_height_{self.type}.h5'),'r')
        std = h5py.File(os.path.join(self.dataset_path,f'standard_deviation_{self.type}.h5'),'r')
        
        # get weights
        classes = np.digitize(agbd.get("agbd")[idx], self.bins) - 1 # subtract 1 because bins start at 1
        weights = self.weights[classes]

        # get ground truth
        gt = (agbd.get("agbd")[idx]-self.norm_values["mean_agbd"])/self.norm_values["sigma_agbd"]
        
        # get image
        canopy_height = canopy_height.get("canopy_height")[idx]
        std = std.get("standard_deviation")[idx]
        mask_no_data = canopy_height==255
        canopy_height = (canopy_height-self.norm_values["mean_can"])/self.norm_values["sigma_can"]
        std = (std-self.norm_values["mean_std"])/self.norm_values["sigma_std"]
        
        gt[mask_no_data] = np.nan # mask no data area so that we don't use those parts for loss computing
        
        # include GEDI standard error as ground truth target
        if self.predict_agbd_se:
            gt_se = (agbd_se.get("agbd_se")[idx]-self.norm_values["mean_agbd_se"])/self.norm_values["sigma_agbd_se"]
            gt_se[mask_no_data] = np.nan
            gt = np.stack([gt, gt_se], axis=0)
        
        # include latitude and longitude as input features
        if self.latlon:
            latitude = np.sin(np.pi*lat.get("lat")[idx]/180.0) # sin transformation is sufficient since latitude is not cyclic
            longitude_cos = np.cos(np.pi*lon.get("lon")[idx]/180.0)
            longitude_sin = np.sin(np.pi*lon.get("lon")[idx]/180.0)
            
            if self.include_std:
                img = np.stack([canopy_height, std, latitude, longitude_cos, longitude_sin], axis=0)
            else:
                img = np.stack([canopy_height, latitude, longitude_cos, longitude_sin], axis=0)
        
        elif self.lat:
            latitude = np.sin(np.pi*lat.get("lat")[idx]/180.0) # sin transformation is sufficient since latitude is not cyclic
            if self.include_std:
                img = np.stack([canopy_height, std, latitude], axis=0)
            else:
                img = np.stack([canopy_height, latitude], axis=0)
            
        else:
            if self.include_std:
                img = np.stack([canopy_height, std], axis=0)
            else:
                img = np.expand_dims(canopy_height, axis=0)
        
        return img, gt, weights

"""
# Not useful anymore (was used for a previous predict function)
class Sentinel2Tile(Dataset):
    def __init__(self, tile_path, dataset_path, latlon=True, include_std=True, normalize_input=True):
        self.tile_path = tile_path
        self.dataset_path = dataset_path
        self.latlon = latlon
        self.include_std = include_std
        if not os.path.exists(self.tile_path):
            raise FileNotFoundError('The folder {} does not exist.'.format(self.tile_path))
        
        if not os.path.exists(os.path.join(self.dataset_path,'normalization_values.csv')):
            raise FileNotFoundError('The file {} does not exist.'.format("normalization_values.csv"))
        
        with open(os.path.join(self.dataset_path,'normalization_values.csv'), mode='r') as infile:
            reader = csv.reader(infile)
            self.norm_values = {rows[0]:float(rows[1]) for rows in reader}
        
        if not normalize_input:
            self.norm_values["mean_can"]=0.0
            self.norm_values["sigma_can"]=1.0
        
        tile = h5py.File(self.tile_path, 'r')
        if len(tile.get("lat").shape)==4:
            self.total = tile.get("lat").shape[0]*tile.get("lat").shape[1]
        elif len(tile.get("lat").shape)==3:
            self.total = tile.get("lat").shape[0]
        else:
            raise ValueError('The shape of the tile file is not good')
        self.patch_size = tile.get("lat").shape[-1]
        self.stride =  tile.get("stride")[0]
        tile.close()
        
    def __len__(self):
        return self.total
    
    def __getitem__(self, idx):
        # c.f. https://github.com/pytorch/pytorch/issues/11929
        # "the actual file opening has to happen inside of the__getitem__function of the Dataset wrapper"   
        self.lat = h5py.File(self.tile_path, 'r').get("lat")[:].reshape(-1, self.patch_size, self.patch_size)[idx]
        self.lon = h5py.File(self.tile_path, 'r').get("lon")[:].reshape(-1, self.patch_size, self.patch_size)[idx]
        self.canopy_height = h5py.File(self.tile_path, 'r').get("canopy_height")[:].reshape(-1, self.patch_size, self.patch_size)[idx]
        self.std = h5py.File(self.tile_path, 'r').get("standard_deviation")[:].reshape(-1, self.patch_size, self.patch_size)[idx]
        
        # get image
        mask_no_data = self.canopy_height == 255
        canopy_height = (self.canopy_height-self.norm_values["mean_can"])/self.norm_values["sigma_can"]
        std = (self.std-self.norm_values["mean_std"])/self.norm_values["sigma_std"]
        # include latitude and longitude as input features
        if self.latlon:
            latitude = np.sin(np.pi*self.lat/180.0) # sin transformation is sufficient since latitude is not cyclic
            longitude_cos = np.cos(np.pi*self.lon/180.0)
            longitude_sin = np.sin(np.pi*self.lon/180.0)
            
            if self.include_std:
                img = np.stack([canopy_height, std, latitude, longitude_cos, longitude_sin], axis=0)
            else:
                img = np.stack([canopy_height, latitude, longitude_cos, longitude_sin], axis=0)
            
        else:
            if self.include_std:
                img = np.stack([canopy_height, std], axis=0)
            else:
                img = np.expand_dims(canopy_height, axis=0)
    
        return img, self.lat, self.lon, mask_no_data
""" 
