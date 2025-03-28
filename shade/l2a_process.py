import sys
import os
from pathlib import Path
from multiprocessing import Pool

L2A_BIN = './Sen2Cor-02.11.00-Linux64/bin/L2A_Process'

if len(sys.argv) != 3:
    print('Usage: python l2a_process.py INPUT_DIR OUTPUT_DIR')
    sys.exit(1)


def process_tile(safe_dir):
    print(f'[Processing] {safe_dir}')
    command = f'{L2A_BIN} {safe_dir} --output_dir {sys.argv[2]}'
    if os.system(command) != 0:
        print(f'[Failed] {safe_dir}')


safe_dirs = [str(p) for p in Path(sys.argv[1]).glob('**/*MSIL1C*.SAFE')]

print(f'Processing {len(safe_dirs)} tiles using {os.cpu_count()} threads')

with Pool(8) as pool:
    pool.map(process_tile, safe_dirs)

print('Done.')
