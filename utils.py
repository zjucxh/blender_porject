import numpy as np
import bpy
import os
from scipy.spatial.transform import Rotation as R

# Load obj file
def load_obj(filename, tex_coords=False):
    vertices = []
    faces = []
    uvs = []
    faces_uv = []

    with open(filename, 'r') as fp:
        for line in fp:
            line_split = line.split()
            
            if not line_split:
                continue

            elif tex_coords and line_split[0] == 'vt':
                uvs.append([line_split[1], line_split[2]])

            elif line_split[0] == 'v':
                vertices.append([line_split[1], line_split[2], line_split[3]])

            elif line_split[0] == 'f':
                vertex_indices = [s.split("/")[0] for s in line_split[1:]]
                faces.append(vertex_indices)

                if tex_coords:
                    uv_indices = [s.split("/")[1] for s in line_split[1:]]
                    faces_uv.append(uv_indices)

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32) - 1

    if tex_coords:
        uvs = np.array(uvs, dtype=np.float32)
        faces_uv = np.array(faces_uv, dtype=np.int32) - 1
        return vertices, faces, uvs, faces_uv

    return vertices, faces
# Load cmu smpl poses 
def load_cmu(pose_path:str):
    """
    Load cmu pose data from npz file.
    
    Args:
        pose_path (str): Path to the pose file.
        
    Returns:
        np.ndarray: Array of poses.
    """
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"Pose file not found: {pose_path}")

    # load npz 
    animation = np.load(pose_path, allow_pickle=True)
    keys = list(animation.keys())
    for key in keys:
        print(f"{key}: {animation[key].shape}") 
    # print mocap_framerate
    print(f"mocap_framerate: {animation['mocap_frame_rate']}")
    return animation

def separate_arms(poses, angle=20, left_arm=17, right_arm=16):
    num_joints = poses.shape[-1] //3

    poses = poses.reshape((-1, num_joints, 3))
    rot = R.from_euler('z', -angle, degrees=True)
    poses[:, left_arm] = (rot * R.from_rotvec(poses[:, left_arm])).as_rotvec()
    rot = R.from_euler('z', angle, degrees=True)
    poses[:, right_arm] = (rot * R.from_rotvec(poses[:, right_arm])).as_rotvec()

    poses[:, 23] *= 0.1
    poses[:, 22] *= 0.1

    return poses.reshape((poses.shape[0], -1))

def finite_diff(x, h, diff=1):
    if diff == 0:
        return x

    v = np.zeros(x.shape, dtype=x.dtype)
    v[1:] = (x[1:] - x[0:-1]) / h

    return finite_diff(v, h, diff-1)

def load_motion(path):
    motion = np.load(path, mmap_mode='r')

    reduce_factor = int(motion['mocap_framerate'] // 30)
    pose = motion['poses'][::reduce_factor, :72]
    trans = motion['trans'][::reduce_factor, :]
    betas = motion['betas'][:10]  # smpl betas
    separate_arms(pose)

    # Swap axes
    swap_rotation = R.from_euler('zx', [-90, 270], degrees=True)
    root_rot = R.from_rotvec(pose[:, :3])
    pose[:, :3] = (swap_rotation * root_rot).as_rotvec()
    trans = swap_rotation.apply(trans)

    # Center model in first frame
    trans = trans - trans[0] 

    # Compute velocities
    trans_vel = finite_diff(trans, 1 / 30)

    return pose.astype(np.float32), betas.astype(np.float32), trans.astype(np.float32), trans_vel.astype(np.float32)
# Check CMU_SNUG data integrity
# TODO align motion data with obj file. SNUG data samples motion data with reduce_factor=framerate//30
def check_cmu_snug_data(data_dir:str="/home/cxh/mnt/cxh/Documents/CMU_SNUG"):
    """
    Check the integrity of CMU SNUG data.
    
    Args:
        data_dir (str): Directory containing the CMU SNUG data.
        
    Returns:
        bool: True if data is valid, False otherwise.
    """
    if not os.path.exists(data_dir):
        print(f"Data directory does not exist: {data_dir}")
        return False

    # walk through the directory get .npz file
    for dirs, dir_name, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npz'):
                print(f' dirs : {dirs}')
                print(f' file name : {file}')
                motion_path = os.path.join(dirs, file) # cmu npz motion data path
                obj_dir = os.path.join(dirs, file.split('_')[1])
                # body obj file path 
                body_obj_path = os.path.join(obj_dir, 'body_{0:0>4}.obj'.format(0))
                try:
                    poses, betas, trans, trans_vel = load_motion(motion_path)
                    pose_len = poses.shape[0]
                    print(f"Pose length: {poses.shape[0]}")
                    for i in range(pose_len):
                        # load body and tshirt obj file
                        body_obj_path = os.path.join(obj_dir, 'body_{0:0>4}.obj'.format(i))
                        tshirt_obj_path = os.path.join(obj_dir, 'tshirt_{0:0>4}.obj'.format(i))
                        # Load obj file
                        body_v, body_f = load_obj(body_obj_path)
                        tshirt_v, tshirt_f = load_obj(tshirt_obj_path)
                        print(f"Loaded body and tshirt obj files for frame {i}")

                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    return False 
    return True

if __name__ == "__main__":
    # check cmu snug data integrity
    data_dir = "/home/cxh/mnt/cxh/Documents/CMU_SNUG"
    check_cmu_snug_data(data_dir)
    print('Done')     