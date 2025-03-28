from pathlib import Path
import pickle
from time import time
import sys
import warnings

import numpy as np
import rasterio
import rasterio.transform
import rasterio.warp
from rasterio.warp import reproject, Resampling
from tqdm import tqdm

from utils import read_sentinel2_bands

TILE_SIZE = 10980
WINDOW_SIZE = 5


def get_veg_indices(img):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'invalid value')
        ndvi = (img[7, :, :] - img[3, :, :]) / (img[7, :, :] + img[3, :, :])
        grvi = (img[2, :, :] - img[3, :, :]) / (img[2, :, :] + img[3, :, :])
        rvi = img[7, :, :] / img[3, :, :]
        gndvi = (img[7, :, :] - img[2, :, :]) / (img[7, :, :] + img[2, :, :])
        ndmi = (img[7, :, :] - img[11, :, :]) / (img[7, :, :] + img[11, :, :])
        
    out = np.stack([
        ndvi,
        grvi,
        rvi,
        gndvi,
        ndmi
    ])
    out[:, (img == 0.).any(axis=0)] = 0.
    return out


def predict_tile(tile_name, tiles_dir, cocoa_map_dir, model_path, out_dir):
    safe_paths = list(Path(tiles_dir).glob(f'**/*{tile_name}*.SAFE'))
    dummy_tile_file = rasterio.open(next(safe_paths[0].glob('**/*10m.jp2')))
    
    cocoa_file = rasterio.open(Path(cocoa_map_dir) / f'T{tile_name}/average_output_predictions.tif')
    cocoa_map = reproject(
        cocoa_file.read(cocoa_file.indexes),
        np.zeros((1, TILE_SIZE, TILE_SIZE), dtype=np.float32),
        src_transform=cocoa_file.transform,
        src_crs=cocoa_file.crs,
        src_nodata=cocoa_file.nodata,
        dst_transform=dummy_tile_file.transform,
        dst_crs=dummy_tile_file.crs,
        dst_nodata=dummy_tile_file.nodata,
        resampling=Resampling.nearest
    )[0]
    cocoa_mask = ~np.isnan(cocoa_map[0]) & (cocoa_map[0] > 0.65)
    del cocoa_map

    with open(model_path, 'rb') as fh:
        model = pickle.load(fh)

    h = WINDOW_SIZE // 2
    out = np.full((TILE_SIZE - 2 * h, TILE_SIZE - 2 * h), 0., dtype=np.float16)
    count = np.full_like(out, 0, dtype=np.uint8)

    # iterate through all tile locations
    print(f'Processing {len(safe_paths)} tiles for {tile_name}...')
    for safe_path in tqdm(safe_paths):
        try:
            tile = read_sentinel2_bands(safe_path)[0]
        except:
           continue
        tile, cld = tile[:-1], tile[-1:]
        tile = np.concatenate([tile, get_veg_indices(tile), cld], axis=0)
        
        # create sliding window view on tile
        win_view = np.lib.stride_tricks.sliding_window_view(
            tile, (WINDOW_SIZE, WINDOW_SIZE), axis=(1, 2))

        valid_mask = \
            cocoa_mask[h:-h, h:-h] & \
            (win_view[:12] != 0).all(axis=(0, -2, -1)) & \
            (win_view[-1] == 0).all(axis=(-2, -1))

        del tile
        prediction = model.predict(win_view[:-1, valid_mask].transpose(1, 0, 2, 3).reshape(-1, model.n_features_in_))
        prediction = np.clip(prediction, 0., 1.).astype(np.float16)

        # update running mean in `out` array
        out[valid_mask] = (out[valid_mask] * count[valid_mask] + prediction) / (count[valid_mask] + 1)
        count[valid_mask] += 1

        del win_view, valid_mask, prediction

    out[count == 0] = np.nan
    out = np.pad(out, ((h, h), (h, h)), constant_values=np.nan)[None]

    print('Writing result...')
    with rasterio.Env():
        s2_profile = dummy_tile_file.profile
        s2_profile.update(
            driver='GTiff',
            count=out.shape[0],
            height=out.shape[1],
            width=out.shape[2],
            tiled=False,
            compress='deflate',
            dtype=np.float16,
            nodata=np.nan
        )
        with rasterio.open(Path(out_dir) / f'SC_{tile_name}.tif', 'w', **s2_profile) as f:
            f.write(out)


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print(f'Usage: predict_gbr_tile_batched.py TILE_NAME TILES_DIR COCOA_MAP_DIR MODEL_PATH OUT_DIR')
        sys.exit(1)
    
    predict_tile(*sys.argv[1:])
