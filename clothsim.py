import blenderproc as bproc
import os
import bpy
import cv2
import numpy as np
from types import SimpleNamespace as SN
from mathutils import Matrix, Vector, Quaternion, Euler

# SMPL body model
class SMPLModel():
    def __init__(self):
        self.kintree = {
            -1: (-1, 'root'),
            0: (-1, 'Pelvis'),
            1: (0, 'L_Hip'),
            2: (0, 'R_Hip'),
            3: (0, 'Spine1'),
            4: (1, 'L_Knee'),
            5: (2, 'R_Knee'),
            6: (3, 'Spine2'),
            7: (4, 'L_Ankle'),
            8: (5, 'R_Ankle'),
            9: (6, 'Spine3'),
            10: (7, 'L_Foot'),
            11: (8, 'R_Foot'),
            12: (9, 'Neck'),
            13: (9, 'L_Collar'),
            14: (9, 'R_Collar'),
            15: (12, 'Head'),
            16: (13, 'L_Shoulder'),
            17: (14, 'R_Shoulder'),
            18: (16, 'L_Elbow'),
            19: (17, 'R_Elbow'),
            20: (18, 'L_Wrist'),
            21: (19, 'R_Wrist'),
            22: (20, 'L_Hand'),
            23: (21, 'R_Hand')
        }
        self.n_bones = 24 # number of bones
        self.gender = 'm' #or 'f' 
        self.betas = np.zeros(10, dtype=np.float32)  # smpl shape parameters
        self.pose = np.zeros(72, dtype=np.float32)  # smpl pose parameters 
        self.pose[66:72] = 0.0 # rest hand

        # Load fbx basic model
        bpy.ops.import_scene.fbx(filepath=os.path.join('assets/model', 
                                                       f'basicModel_{self.gender}_lbs_10_207_0_v1.0.2.fbx'),
                                                       axis_forward='Y', axis_up='Z', global_scale=100)
        
        #bpy.ops.wm.obj_import
        # Get the armature object
        self.obname = f'{self.gender}_avg'
        self.body = bpy.data.objects[self.obname]
        #self.body.data.use_auto_smooth = False
        self.body.data.shape_keys.animation_data_clear()
        self.armature = bpy.data.objects['Armature']
        self.armature.scale = [100, 100, 100] # Scale for accurate simulation

        # Load tshirt obj file
        #bpy.ops.wm.obj_import(filepath=os.path.join('assets/', 'tshirt.obj'))
                                    #axis_forward='-Z', axis_up='Y')
        #self.tshirt = bpy.data.objects['tshirt']
        #self.tshirt.scale = [.01, .01, .01] 
        bpy.context.view_layer.update() # Update the scene

    def deselect(self):
        for o in bpy.data.objects.values():
            o.select_set(False)
        bpy.context.view_layer.objects.active = None

    def bone_name(self, i, bodyname='f_avg'):
        """
        Get the name of the bone by index.
        """
        if i < 0 or i >= self.n_bones:
            raise ValueError(f"Bone index {i} is out of range.")
        return (self.obname + '_' + self.kintree[i][1]) 

    # rodrigues transformation. input rotation vector, return rotation matrix
    def rodrigues(self, rotvec):
        """
        Convert a rotation vector to a rotation matrix using OpenCV's Rodrigues function.
        """
        r, _ = cv2.Rodrigues(rotvec)
        return r
    
    # pose to rotation matrix and pose blend shapes
    def rodrigues2bshapes(self,  pose):
        rod_rots = np.asarray(pose, dtype=np.float32).reshape(-1, 3)
        mat_rots = np.asarray([self.rodrigues(rod_rot) for rod_rot in rod_rots], dtype=np.float32)
        bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel() for mat_rot in mat_rots[1:]])
        return (mat_rots, bshapes)
    
    # Apply shape and pose to frame in blender
    def apply_shape_pose(self, beta, pose, frame):
        # set beta parameter ranges from -10 to 10
        for k in self.body.data.shape_keys.key_blocks.keys():
            self.body.data.shape_keys.key_blocks[k].slider_min = -10
            self.body.data.shape_keys.key_blocks[k].slider_max = 10
            bpy.data.shape_keys['Key'].key_blocks[k].slider_max = 10
            bpy.data.shape_keys['Key'].key_blocks[k].slider_min = -10
        mpose = np.zeros(shape=(self.n_bones, 3, 3), dtype=np.float32)
        pose = pose.reshape(-1, 3)
        _, bshapes = self.rodrigues2bshapes(pose)
        for i, p in enumerate(pose):
            if i <= 1:
                continue
            mrot = self.rodrigues(p)
            mpose[i] = mrot
            bone = self.armature.pose.bones[self.bone_name(i, bodyname=f'{self.gender}_avg')]
            if i == 0:
                bone.location = [0, 0, 0]
            bone.rotation_quaternion = Matrix(mrot).to_quaternion()
            bone.keyframe_insert('rotation_quaternion', frame=frame)
        for ibeta, val in enumerate(beta):
            #print("Setting shape key {0} to value {1} at frame {2}".format(ibeta, val, frame))
            self.body.data.shape_keys.key_blocks['Shape{0:0>3}'.format(ibeta)].value = val
            self.body.data.shape_keys.key_blocks['Shape{0:0>3}'.format(ibeta)].keyframe_insert('value',frame=frame)

        # Apply bshape to frame
        for ibshape, val in enumerate(bshapes):
            #print("Setting shape blend shape {0} to value {1} at frame {2}".format(ibshape, val, frame))
            self.body.data.shape_keys.key_blocks['Pose{0:0>3}'.format(ibshape)].value = val
            self.body.data.shape_keys.key_blocks['Pose{0:0>3}'.format(ibshape)].keyframe_insert('value', frame=frame)
        bpy.context.view_layer.update()


    # Load smplh poses
    def load_cmu(self, pose_path:str):
        """
        Load CMU pose data from file.

        Args:
            pose_path (str): Path to the pose file.

        Returns:
            dict: Dictionary containing pose data.
        """
        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"Pose file not found: {pose_path}")

        # load npz 

        animation = np.load(pose_path, allow_pickle=True)
        #keys = list(animation.keys())
        #for key in keys:
        #    print(f"{key}: {animation[key].shape}") 
        # print mocap_framerate
        #print(f"mocap_framerate: {animation['mocap_framerate']}")
        return animation
    
    def visualize(self, npz_data:str):
        """
        Visualize the SMPLH pose data in Blender.

        Args:
            npz_data (str): Path to the npz file containing pose data.
        """
        poses = self.load_cmu(npz_data)
        betas = poses['betas'][:10]  # shape parameters
        pose = poses['poses'][:,:72]  # pose parameters
        pose[:,66:72] = 0.0  # rest hand pose
        # apply shape and pose to frame in blender
        for i, p in enumerate(pose):
            print(f"Applying shape and pose for frame {i}")
            # apply shape and pose
            self.apply_shape_pose(betas, p, frame=i)

    # TODO cloth simulation for imported SMPL model
    def simulate(self):
        raise NotImplementedError("Cloth simulation is not implemented yet. Please implement the simulate method.")

if __name__ == "__main__":
    # initialize BlenderProc
    bproc.init()
    # Create instance of SMPLModel
    smpl_model = SMPLModel()
    smpl_model.visualize('assets/smplh_poses.npz')
    

    