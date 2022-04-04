from myvgn.a import *
from myvgn.simulation import GraspSim
from myvgn.utils.transform import Rotation, Transform
from myvgn.perception import camera_on_sphere, create_tsdf
from myvgn.io import *
import open3d as o3d
from myvgn.grasp import Grasp, Label
import scipy.signal as signal
from tqdm import tqdm
from pathlib import Path

GRASPS_PER_SCENE = 120

def main(root: str, total_grasps: int, gui: bool):
    (root / "scenes").mkdir(parents=True, exist_ok=True)
    
    pbar = tqdm(total=(total_grasps // GRASPS_PER_SCENE)*GRASPS_PER_SCENE)

    #make sim
    sim = GraspSim(
        scene="air",
        object_set="blocks",
        gui=gui,
        seed=777
    )
    finger_depth = sim.gripper.finger_depth  # 0.05
    write_setup(
        root,
        sim.size,
        sim.camera.intrinsic,
        sim.gripper.max_opening_width,
        sim.gripper.finger_depth,
    )
    
    for _ in range(total_grasps // GRASPS_PER_SCENE):
        sim.reset(object_count=1)
        sim.save_state()
        depth_imgs, extrinsics = render_images(sim, 12)
        tsdf = create_tsdf(sim.size, 120, depth_imgs, sim.camera.intrinsic, extrinsics)
        pc = tsdf.get_cloud()
        #o3d.visualization.draw_geometries([pc])
        
        if pc.is_empty():
            print("point cloud empty, skipping scene")
            #continue
        else:
            scene_id = write_sensor_data(root, depth_imgs, extrinsics)

        #try grasp
        for _ in range(GRASPS_PER_SCENE):
            point, normal = sample_grasp_point(pc, finger_depth)
            grasp, label = evaluate_grasp_point(sim, point, normal)
            write_grasp(root, scene_id, grasp, label)
            pbar.update()
    pbar.close()

def sample_grasp_point(point_cloud, finger_depth, eps=0.1):
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    center = np.mean(points, axis=0) #0,0,0
    
    idx = np.random.randint(len(points))
    point, normal = points[idx], normals[idx]
    if normal @ (center - point) > 0:
        normal = -normal
    # #while not ok:
    #     # TODO this could result in an infinite loop, though very unlikely
    #     idx = np.random.randint(len(points))
    #     point, normal = points[idx], normals[idx]
    #     #ok = normal[2] > -0.1  # make sure the normal is poitning upwards

    grasp_depth = np.random.uniform(-eps * finger_depth, (1.0 + eps) * finger_depth)
    point = point + normal * grasp_depth
    return point, normal

def evaluate_grasp_point(sim: GraspSim, pos, normal, num_rotations=6):
    # define initial grasp frame on object surface
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)

    # try to grasp with different yaw angles
    yaws = np.linspace(0.0, np.pi, num_rotations) #only consider pi
    outcomes, widths = [], []
    for yaw in yaws:
        ori = R * Rotation.from_euler("z", yaw)
        sim.restore_state()
        candidate = Grasp(Transform(ori, pos), width=sim.gripper.max_opening_width)
        outcome, width = sim.execute_grasp(candidate, remove=False)
        outcomes.append(outcome)
        widths.append(width)

    # detect mid-point of widest peak of successful yaw angles
    # TODO currently this does not properly handle periodicity
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(float)
    if np.sum(successes):
        peaks, properties = signal.find_peaks(
            x=np.r_[0, successes, 0], height=1, width=1
        )
        idx_of_widest_peak = peaks[np.argmax(properties["widths"])] - 1
        ori = R * Rotation.from_euler("z", yaws[idx_of_widest_peak])
        width = widths[idx_of_widest_peak]

    return Grasp(Transform(ori, pos), width), int(np.max(outcomes))

def render_images(sim, n):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)

    thetas = np.array([
        -np.pi/3, -np.pi/3, -np.pi/3, -np.pi/3, 
        0.05, 0.05, 0.05, 0.05, 
        np.pi/3, np.pi/3, np.pi/3, np.pi/3
    ])
    phis = np.array([
        0, np.pi/2, np.pi, np.pi*3/2, 
        0, np.pi/2, np.pi, np.pi*3/2,
        0, np.pi/2, np.pi, np.pi*3/2,
    ]) + np.pi/6
    for i in range(n):
        r = np.random.uniform(1.6, 2.4) * sim.size

        extrinsic = camera_on_sphere(origin, r, thetas[i], phis[i])
        depth_img = sim.camera.render(extrinsic)[1]

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img

    return depth_imgs, extrinsics

if __name__ == "__main__":
    main(
        root=Path("data/raw/air"),
        total_grasps=1000,
        gui=False
    )