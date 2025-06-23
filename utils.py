import numpy as np
import bpy
import os

# Load smplx poses 
def load_smplx(pose_path:str):
    """
    Load SMPLX pose data from a file.
    
    Args:
        pose_path (str): Path to the pose file.
        
    Returns:
        np.ndarray: Array of poses.
    """
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"Pose file not found: {pose_path}")

    # load npz 
     
    animation = np.load(pose_path, allow_pickle=True)
    # List all keys and shape 
    keys = list(animation.keys())
    for key in keys:
        print(f"{key}: {animation[key].shape}")
    # print mocap_framerate
    print(f"mocap_framerate: {animation['mocap_frame_rate']}") # capture framerate
    print(f'mocap_time_length: {animation["mocap_time_length"]}') # Overall time, sequencial length / mocap_framerate

    return animation

# Load smplh poses
def load_smplh(pose_path:str):
    """
    Load SMPLH pose data from a file.
    
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
    print(f"mocap_framerate: {animation['mocap_framerate']}")
    return animation

# TODO: Convert SMPLH to SMPLX format
def smplh2x(smplh, smplx):
    """
    Convert SMPLH pose data to SMPLX format.
    
    Args:
        smplh (np.ndarray): SMPLH pose data.
        smplx (np.ndarray): SMPLX pose data.
        
    Returns:
        np.ndarray: Converted SMPLX pose data.
    """
    # This is a placeholder for the actual conversion logic
    # You would need to implement the conversion based on your requirements
    # Load smplh npz data 
    trans = smplh['trans']
    gender = smplh['gender']
    mocap_framerate = smplh['mocap_framerate']
    betas = smplh['betas']
    poses = smplh['poses']
    dmpls = smplh['dmpls'] if 'dmpls' in smplh else None
    # print mocap_framerate
    print(f"mocap_framerate: {mocap_framerate}")

    return smplx  # Return the original for now
if __name__ == "__main__":
    # Example usage
    load_smplx('assets/smplx_poses.npz')
    load_smplh('assets/smplh_poses.npz')
    print('Done')     