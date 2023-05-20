import argparse
import collections.abc as collections
from pathlib import Path
from typing import Optional, Union, List

from . import logger
import os


def main(
        output: Path,
        folder_path: Optional[Union[Path, List[str]]],
        pair_num: int = 4):
    
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    image_files.sort()
    names_ref = image_files

    pairs = []
    for i in range(0, len(names_ref)-pair_num):
        for j in range(i+1, i+pair_num+1):
            pairs.append((names_ref[i], names_ref[j]))

    logger.info(f'Found {len(pairs)} pairs.')
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--image_list', type=Path)
    parser.add_argument('--features', type=Path)
    parser.add_argument('--ref_list', type=Path)
    parser.add_argument('--ref_features', type=Path)
    args = parser.parse_args()
    main(**args.__dict__)
