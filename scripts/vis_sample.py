import argparse
from pathlib import Path

import numpy as np
import rospy

from myvgn import vis
from myvgn.dataset import Dataset
from myvgn.grasp import Grasp
from myvgn.utils.transform import Rotation, Transform


def main(dataset, augment=True, num=100):
    rospy.init_node("vgn_vis", anonymous=True)

    dataset = Dataset(dataset, augment=augment)
    
    for i in range(num):
        i = np.random.randint(len(dataset))

        voxel_grid, (label, rotations, width), index = dataset[i]
        grasp = Grasp(Transform(Rotation.from_quat(rotations[0]), index), width)

        vis.clear()
        vis.draw_workspace(40)
        vis.draw_tsdf(voxel_grid, 1.0)
        vis.draw_grasp(grasp, float(label), 40.0 / 6.0)

        rospy.sleep(1.0)


if __name__ == "__main__":
    main(dataset=Path("data/datasets/air"))
