from pathlib import Path
import json
from datetime import datetime
import re
import warnings
from collections import defaultdict
import pickle

import click
import numpy as np
import rasterio
import fiona
from tqdm import tqdm
from rasterio.coords import disjoint_bounds
from rasterio.warp import reproject, transform_bounds
import rasterio.warp
import rasterio.mask
from rasterio.merge import merge
from shapely.geometry import shape
from pyproj import Geod
from affine import Affine
from skimage.measure import block_reduce
from sklearn.ensemble import GradientBoostingRegressor

MIN_COVERAGE = 0.8
N_CLOSEST_IMG = 10
MAX_DIFF_DAYS = 120
SHADE_THRESHOLD = 8.

WINDOW_SIZE = 5
PRED_WINDOW_SIZE = 1
AVERAGE = True
N_ESTIMATORS = 2000
LOSS = 'huber'


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


def process_s2(s2_img):
    s2_img = s2_img[:-1]
    veg_indices = get_veg_indices(s2_img)
    s2_img = np.concatenate([s2_img, veg_indices], axis=0)
    padding = WINDOW_SIZE // 2
    s2_img = np.pad(s2_img, ((0, 0), (padding, padding), (padding, padding)))
    s2_img = np.lib.stride_tricks.sliding_window_view(s2_img, (WINDOW_SIZE, WINDOW_SIZE), axis=(1, 2))
    s2_img = s2_img.transpose((1, 2, 0, 3, 4)).reshape((*s2_img.shape[1:3], -1)).transpose((2, 0, 1))
    return s2_img


def process_gt(gt):
    padding = WINDOW_SIZE // 2
    margin = (WINDOW_SIZE - PRED_WINDOW_SIZE) // 2
    gt = np.pad(gt, ((0, 0), (padding, padding), (padding, padding)))
    gt = np.lib.stride_tricks.sliding_window_view(gt, (WINDOW_SIZE, WINDOW_SIZE), axis=(1, 2))
    gt = gt[0, :, :, margin:-margin if margin != 0 else None, margin:-margin if margin != 0 else None] \
            .reshape((*gt.shape[1:3], -1)).transpose((2, 0, 1))
    
    if AVERAGE:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'Mean of empty slice')
            gt = np.nanmean(gt, axis=0, keepdims=True)
    return gt


def get_area_m2(polygon):
    polygon = shape(polygon['geometry'])
    geod = Geod(ellps="WGS84")
    return abs(geod.geometry_area_perimeter(polygon)[0])


def build_farms_dataset(polygons_path, flight_dates_path, split_file_path, reprojected_path, s2_path):
    farm_polygons = []
    for polygons_file in list(polygons_path.glob('*.shp')) + list(polygons_path.glob('*.geojson')):
        if polygons_file.stem.startswith('.'):
            continue
        with fiona.open(polygons_file) as polygons:
            farm_polygons.extend([(p, polygons.crs, polygons_file) for p in polygons])

    flight_dates = {}
    with open(flight_dates_path) as fh:
        for k, v in json.load(fh).items():
            low, up = [int(flight_id[1:]) for flight_id in k.split('-')]
            date = datetime.strptime(v, '%Y-%m-%d')
            flight_dates.update({f'{k[0]}{flight_no:02}': date for flight_no in range(low, up + 1)})
           
    def get_flight_date(flight_name):
        # CI flights
        match = re.search('_(2021\d+)_', flight_name)
        if match:
            return datetime.strptime(match.groups()[0], '%Y%m%d')
        
        first_flight = re.match(r'((F|P|H)\d*)', flight_name).groups()[0]
        if first_flight in flight_dates:
            return flight_dates[first_flight]
        
        raise IndexError()
        
    # build dataset
    with open(split_file_path) as fh:
        d = json.load(fh)
        train_flight_names = d['train']
        val_flight_names = d.get('val', d['test'])

    F_train, F_val = [], []

    for mode in ('train', 'val'):
        if mode == 'train':
            F_current = F_train
        else:
            F_current = F_val
            
        print(f'Building <{mode}> dataset')

        for farm_polygon, farm_crs, farm_file in tqdm(farm_polygons):
            intersecting_paths = []
            crs, nodata = None, None
            seen = set()

            for img_path in reprojected_path.glob('*.tif'):
                flight_name = img_path.stem[:-6]
                if flight_name not in (train_flight_names if mode == 'train' else val_flight_names) or flight_name in seen:
                    continue
                # only use one S2 tile per flight now, almost no effect anyway
                seen.add(flight_name)

                with rasterio.open(img_path) as fh:
                    if not disjoint_bounds(transform_bounds(farm_crs, fh.crs, *shape(farm_polygon['geometry']).bounds), fh.bounds):
                        if crs is None:
                            crs = fh.crs
                            nodata = fh.nodata
                        intersecting_paths.append(img_path)

            if not len(intersecting_paths):
                continue

            merged, merged_transform = merge(intersecting_paths)
            polygon_mask_hr = ~rasterio.features.geometry_mask(
                [rasterio.warp.transform_geom(farm_crs, crs, farm_polygon['geometry'])], 
                merged.shape[1:], merged_transform
            )

            if not polygon_mask_hr.any():
                continue 

            height_values = merged[0][polygon_mask_hr]

            gt_mask_hr = height_values != nodata
            area = get_area_m2(farm_polygon)
            coverage = (gt_mask_hr.sum() * merged_transform.a**2) / area
            # coverage is defined as the portion of the polygon that is covered by non-nodata values
            if coverage <= MIN_COVERAGE:
                continue

            gsd_factor = round(10. / merged_transform.a)
            merged[merged == nodata] = np.nan
            shade_cover_gt = np.full_like(merged, np.nan)
            shade_cover_gt[~np.isnan(merged)] = (merged[~np.isnan(merged)] > SHADE_THRESHOLD).astype('float32')
            shade_cover_gt = block_reduce(shade_cover_gt, (1, gsd_factor, gsd_factor), np.mean)
            gt_mask_lr = ~np.isnan(shade_cover_gt)[0]

            polygon_mask_lr = ~rasterio.features.geometry_mask(
                [rasterio.warp.transform_geom(farm_crs, crs, farm_polygon['geometry'])], 
                shade_cover_gt.shape[1:],
                Affine(10., 0., merged_transform.c, 0., -10., merged_transform.f)
            )

            s2_date_path_map = defaultdict(list)  # date -> path
            for intersecting_path in intersecting_paths:
                flight_name, tile_name = intersecting_path.stem[:-6], intersecting_path.stem[-5:]
                flight_date = get_flight_date(flight_name)

                for p in list((s2_path / flight_name).glob(f'*_T{tile_name}.tif')):
                    s2_date_path_map[datetime.strptime(p.stem.split('T')[0], '%Y%m%d')].append(p)

            s2_imgs = []  # list[tuple[<img>, <transform>, <date>]]
            for date, s2_paths in s2_date_path_map.items():
                assert len(s2_paths) <= len(intersecting_paths)
                if len(s2_paths) == len(intersecting_paths):
                    # merge all s2 images of that date
                    s2_img, s2_transform = merge(s2_paths)

                    # cloud cover & nonzero check; only look at valid & polygon regions
                    patch = s2_img[:, gt_mask_lr & polygon_mask_lr]
                    if (patch[-1] == 0).all() and (patch[:-1] > 0).all():
                        s2_imgs.append((s2_img, s2_transform, date))

            s2_imgs = [e for e in s2_imgs if abs((e[2] - flight_date).days) <= MAX_DIFF_DAYS]
            s2_imgs.sort(key=lambda e: abs((e[2] - flight_date).days))
            s2_imgs = s2_imgs[:N_CLOSEST_IMG]

            if not len(s2_imgs):
                print(f'[{intersecting_paths[0].stem}] No s2 images found')
                continue

            F_current.append(([img for img, _, _ in s2_imgs], shade_cover_gt, 
                gt_mask_lr & polygon_mask_lr, gt_mask_lr))
            
    return F_train, F_val


def build_pixel_dataset(farms):
    X, y = [], []
    for s2_imgs, shade_cover_gt, mask, _ in farms:
        for s2_img in s2_imgs:
            s2_img = process_s2(s2_img)
            X.append(s2_img[:, mask].T)
            shade = process_gt(shade_cover_gt)
            y.append(shade[:, mask])

    X, y = np.concatenate(X), np.concatenate(y, axis=1).T
    non_nan = ~np.isnan(y).any(axis=1)
    X, y = X[non_nan], y[non_nan]

    assert not np.isnan(X).any() and not np.isnan(y).any()
    return X, y


@click.command()
@click.option(
    '--reprojected-path',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the reprojected flight dataset."
)
@click.option(
    '--s2-path',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the reprojected optical Sentinel-2 dataset."
)
@click.option(
    '--polygons-path',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the polygons dataset."
)
@click.option(
    '--flight-dates-path',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the flight dates JSON file."
)
@click.option(
    '--split-file-path',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the split train-test JSON file."
)
@click.option(
    '--save-path',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Save path of the trained model'
)
def main(reprojected_path, s2_path, polygons_path, flight_dates_path, split_file_path, save_path):
    F_train, F_val = build_farms_dataset(
        polygons_path, flight_dates_path, split_file_path, reprojected_path, s2_path)
    print(f'Found {len(F_train)} train and {len(F_val)} val farms')
    
    X_train, y_train = build_pixel_dataset(F_train)
    X_val, y_val = build_pixel_dataset(F_val)
    
    model = GradientBoostingRegressor(n_estimators=N_ESTIMATORS, loss=LOSS, verbose=1)
    model.fit(X_train, y_train)
    
    with open(save_path, 'wb') as fh:
        pickle.dump(model, fh)
        
    # eval on pixel level
    y_pred = model.predict(X_val).flatten()
    mae = np.abs(y_pred - y_val.flatten()).mean()
    mse = ((y_pred - y_val.flatten())**2).mean()
    print('===== pixel level validation (one image) =====')
    print(f'MAE: {mae}\nMSE:{mse}')
    
    # eval on farm level -- average multi time
    preds, gts, outs, gts_farm = [], [], [], []
    for s2_imgs, shade_cover_gt, mask, _ in tqdm(F_val):
        if mask.sum() == 0:
            continue

        preds_t, gts_t = [], []
        for s2_img in s2_imgs:
            s2_img = process_s2(s2_img)

            pred = model.predict(s2_img.reshape(s2_img.shape[0], -1).T).reshape((-1, 1))
            pred = pred.reshape(*mask.shape, int(np.sqrt(pred.shape[1])), int(np.sqrt(pred.shape[1])))
            pred[~mask] = np.nan

            preds_t.append(pred[mask].mean())
            gts_t.append(shade_cover_gt[0][mask].mean())

        # average over s2 images
        preds.append(np.mean(preds_t, axis=0))
        gts.append(np.mean(gts_t, axis=0))

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    mae = np.abs(gts - preds).mean()
    mse = ((gts - preds)**2).mean()
    print('===== farm level validation (multiple images) =====')
    print(f'MAE: {mae}\nMSE:{mse}')


if __name__ == '__main__':
    main()
