import argparse
from pathlib import Path

from myvgn.detection import VGN
from myvgn.experiments import clutter_removal
import rospy

def main(
    model: Path, 
    rviz: bool, 
    logdir: Path, 
    description: str,
    scene: str
):

    # if args.rviz or str(args.model) == "gpd":
    #     import rospy

    #     rospy.init_node("sim_grasp", anonymous=True)

    # if str(args.model) == "gpd":
    #     from vgn.baselines import GPD

    #     grasp_planner = GPD()
    # else:
    grasp_planner = VGN(model, rviz=rviz)
    rospy.init_node("sim_grasp", anonymous=True)
    clutter_removal.run(
        grasp_plan_fn=grasp_planner,
        logdir=logdir,
        description=description,
        scene=scene,
        object_set="blocks",
        sim_gui=True,
        rviz=rviz
    )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=Path, required=True)
    # parser.add_argument("--logdir", type=Path, default="data/experiments")
    # parser.add_argument("--description", type=str, default="")
    # parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    # parser.add_argument("--object-set", type=str, default="blocks")
    # parser.add_argument("--num-objects", type=int, default=5)
    # parser.add_argument("--num-rounds", type=int, default=100)
    # parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--sim-gui", action="store_true")
    # parser.add_argument("--rviz", action="store_true")
    # args = parser.parse_args()
    main(
        model=Path("data/models/vgn_conv.pt"), 
        rviz=True, 
        logdir=Path("data/experiments"), 
        description="",
        scene="air"
    )
