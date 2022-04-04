import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

from myvgn.io import *
from myvgn.perception import *

RESOLUTION = 40


def main(raw: Path, dataset: Path):
    # create directory of new dataset
    (dataset / "scenes").mkdir(parents=True)

    # load setup information
    size, intrinsic, _, finger_depth = read_setup(raw)
    assert np.isclose(size, 6.0 * finger_depth)
    voxel_size = size / RESOLUTION

    # create df
    df = read_df(raw)
    df["x"] /= voxel_size
    df["y"] /= voxel_size
    df["z"] /= voxel_size
    df["width"] /= voxel_size
    df = df.rename(columns={"x": "i", "y": "j", "z": "k"})
    write_df(df, dataset)

    # create tsdfs
    for f in tqdm(list((raw / "scenes").iterdir())):
        if f.suffix != ".npz":
            continue
        depth_imgs, extrinsics = read_sensor_data(raw, f.stem)
        tsdf = create_tsdf(size, RESOLUTION, depth_imgs, intrinsic, extrinsics)
        grid = tsdf.get_grid()
        write_voxel_grid(dataset, f.stem, grid)


if __name__ == "__main__":
    main(raw=Path("data/raw/air"), dataset=Path("data/datasets/air"))
