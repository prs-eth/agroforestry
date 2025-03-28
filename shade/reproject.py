"""
Script for building the reprojected flight dataset.
"""

import numpy as np
from pathlib import Path

import click
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from shapely.geometry import shape
import fiona
from affine import Affine
from tqdm import tqdm

from utils import read_sentinel2_bands

MARGIN = 60  # m
S2_GSD = 10


@click.command()
@click.option(
    '--veg-height-path',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the vegetation height files"
)
@click.option(
    '--s2-path',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the S2 .SAFE files"
)
@click.option(
    '--s2-tiles-shp-path',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to an shp file delineating S2 tile boundaries"
)
@click.option(
    '--reprojected-s2-path',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Output path of reprojected S2 files"
)
@click.option(
    '--reprojected-veg-path',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Output path of reprojected veg height files"
)
def main(veg_height_path, s2_path, s2_tiles_shp_path, reprojected_s2_path, reprojected_veg_path):
    s2_safe_paths = list(s2_path.glob('*.SAFE'))

    with fiona.open(s2_tiles_shp_path) as s2_features_file:
        s2_bounds = {f['properties']['Name']: shape(f['geometry']).bounds for f in s2_features_file}
        s2_features_crs = s2_features_file.crs

    for flight_dir in tqdm(list(veg_height_path.iterdir())):
        with rasterio.open(str(next(flight_dir.glob('*.tif')))) as flight_file:
            # assert flight_file.crs == 'EPSG:3857'
            flight_bounds = flight_file.bounds
            # leave some space around flight
            flight_bounds = [flight_bounds[0] - MARGIN, flight_bounds[1] - MARGIN,
                             flight_bounds[2] + MARGIN, flight_bounds[3] + MARGIN]

            tiles = []
            for name, bounds in s2_bounds.items():
                # transformed flight bounds
                left, bottom, right, top = transform_bounds(flight_file.crs, s2_features_crs, *flight_bounds)
                if bounds[0] < left and bounds[1] < bottom and bounds[2] > right and bounds[3] > top:
                    s2_path = next(p for p in s2_safe_paths if f'T{name}' in str(p))
                    with rasterio.open(next(s2_path.glob('**/*10m.jp2'))) as s2_file:
                        tiles.append((name, s2_file.profile))

            for tile_name, s2_profile in tiles:
                # find bounds that agree with optical grid, so no resampling is performed during merge
                s2_anchor = s2_profile['transform'].c, s2_profile['transform'].f  # arbitrary point on s2 grid
                flight_bounds_t = transform_bounds(flight_file.crs, s2_profile['crs'], *flight_bounds)
                flight_bounds_t = [
                    s2_anchor[0] - ((s2_anchor[0] - flight_bounds_t[0]) // S2_GSD) * S2_GSD,
                    s2_anchor[1] - ((s2_anchor[1] - flight_bounds_t[1]) // S2_GSD) * S2_GSD,
                    s2_anchor[0] - ((s2_anchor[0] - flight_bounds_t[2]) // S2_GSD) * S2_GSD,
                    s2_anchor[1] - ((s2_anchor[1] - flight_bounds_t[3]) // S2_GSD) * S2_GSD
                ]

                # process all acquisition dates for selected tiles
                for safe_path in (p for p in s2_safe_paths if f'T{tile_name}' in str(p)):
                    array, transform = read_sentinel2_bands(str(safe_path), bounds=flight_bounds_t)

                    if (array == 0.).all(0).any():
                        # at least one pixel is all zeros
                        continue

                    with rasterio.Env():
                        s2_profile.update(
                            driver='GTiff',
                            transform=transform,
                            count=array.shape[0],
                            height=array.shape[1],
                            width=array.shape[2],
                            tiled=False,
                            compress='deflate'
                        )
                        save_dir = reprojected_s2_path / flight_dir.stem
                        save_dir.mkdir(exist_ok=True, parents=True)
                        parts = safe_path.stem.split('_')
                        with rasterio.open(str(save_dir / f'{parts[2]}_{parts[5]}.tif'), 'w', **s2_profile) as f:
                            f.write(array)

                # reproject flight to quads CRS, use resolution that is a divider of the quads resolution
                gsd_factor = int(S2_GSD / flight_file.res[0])
                new_gsd = S2_GSD / gsd_factor
                dst_transform = Affine(new_gsd, 0., transform.c, 0., -new_gsd, transform.f)
                flight = reproject(
                    flight_file.read(flight_file.indexes),
                    np.zeros((flight_file.count, array.shape[1] * gsd_factor, array.shape[2] * gsd_factor), dtype=np.float32),
                    src_transform=flight_file.transform,
                    src_crs=flight_file.crs,
                    src_nodata=flight_file.nodata,
                    dst_transform=dst_transform,
                    dst_crs=s2_profile['crs'],
                    dst_nodata=flight_file.nodata,
                    resampling=Resampling.bilinear
                )[0]

                with rasterio.Env():
                    profile = flight_file.profile
                    profile.update(
                        transform=dst_transform,
                        height=flight.shape[1],
                        width=flight.shape[2],
                        crs=s2_profile['crs'],
                        compress='deflate'
                    )
                    save_dir = reprojected_veg_path
                    save_dir.mkdir(exist_ok=True, parents=True)
                    with rasterio.open(str(save_dir / f'{flight_dir.stem}_{tile_name}.tif'), 'w', **profile) as f:
                        f.write(flight)


if __name__ == "__main__":
    main()
