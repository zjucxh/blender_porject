import numpy as np
import bpy
import os

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
                    animation = np.load(motion_path, allow_pickle=True)
                    pose_len = animation['poses'].shape[0]
                    print(f"Pose length: {pose_len}")
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