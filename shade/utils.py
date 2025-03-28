import os
from pathlib import Path
import numpy as np
from skimage.transform import resize
from rasterio.merge import merge


def sort_s2_band_arrays(band_arrays, channels_last=True):
    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'CLD']
    out_arr = []
    for b in bands:
        out_arr.append(band_arrays[b])
    out_arr = np.concatenate(out_arr, axis=0)
    if channels_last:
        out_arr = np.moveaxis(out_arr, source=0, destination=-1)
    return out_arr


def get_tile_info(refDataset):
    tile_info = {'projection': refDataset.GetProjection(), 'geotransform': refDataset.GetGeoTransform(),
                 'width': refDataset.RasterXSize, 'height': refDataset.RasterYSize}
    return tile_info


def read_sentinel2_bands(data_path, channels_last=False, bounds=None):
    bands10m = ['B02', 'B03', 'B04', 'B08']
    bands20m = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'SCL']
    bands60m = ['B01', 'B09']  # 'B10' is missing in 2A, exists only in 1C

    bands_dir = {10: {'band_names': bands10m, 'subdir': 'R10m', 'scale': 1},
                 20: {'band_names': bands20m, 'subdir': 'R20m', 'scale': 2},
                 60: {'band_names': bands60m, 'subdir': 'R60m', 'scale': 6}}

    band_arrays = {}
    transform = None
    for res in bands_dir.keys():
        bands_dir[res]['band_data_list'] = []
        for i in range(len(bands_dir[res]['band_names'])):
            band_name = bands_dir[res]['band_names'][i]
            path_img_data = next(Path(data_path).glob(f'**/*{band_name}_{res}m.jp2'))
            path_band = os.path.join(data_path, path_img_data)

            band_data, t = merge([path_band], bounds)
            if transform is None:
                transform = t
            band_arrays[band_name] = band_data

    path_img_data = next(p for p in Path(data_path).glob(f'**/MSK_CLDPRB_20m.jp2'))
    path_band = os.path.join(data_path, path_img_data)
    band_arrays['CLD'], _ = merge([path_band], bounds)

    target_shape = band_arrays['B02'].shape
    for band_name in band_arrays:
        band_array = band_arrays[band_name]
        if band_array.shape != target_shape:
            band_arrays[band_name] = \
                resize(band_array, target_shape, mode='reflect', order=0, preserve_range=True).astype(np.uint16)

    image_array = sort_s2_band_arrays(band_arrays=band_arrays, channels_last=channels_last)
    return image_array, transform
