import argparse
import bpy
import json
import numpy as np
import os
import sys
from mathutils import Matrix, Vector, Quaternion, Euler
from types import SimpleNamespace as SN

# global variable: poses
global_pose_set_dir = '/home/cxh/Downloads/dataset/PoseSet/'
global_pose_shape_set_dir = '/home/cxh/Downloads/dataset/PoseShapeSet/'

# poses selected from MoLab AMASS dataset
global_pose = {
    0: "catching_and_throwing_poses",
    1: "jumping_poses",
    2: "kicking_poses",
    3: "knocking_poses",
    4: "lifting_heavy_poses",
    5: "lifting_light_poses",
    6: "motorcycle_poses",
    7: "normal_jog_poses",
    8: "normal_walk_poses",
    9: "scamper_poses",
    10: "sitting2_poses",
    11: "sitting_poses",
    12: "throwing_hard_poses",
    13: "treadmill_jog_poses",
    14: "treadmill_norm_poses"
}


class BodyModel():
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
        self.n_bones = 24
        self.gender = 'm'  # or 'f'
        self.betas = np.zeros(10, dtype=np.float32)  # length of beta = 10
        self.poses = np.zeros(72, dtype=np.float32)  # length of theta = 72
        # reset hand
        self.poses[66:72] = 0.0
        # Load basic model
        bpy.ops.import_scene.fbx(
            filepath=os.path.join('model',
                                  f'basicModel_{self.gender}_lbs_10_207_0_v1.0.2.fbx'),
            axis_forward='Y', axis_up='Z', global_scale=1
        )
        self.obname = f'{self.gender}_avg'
        self.body = bpy.data.objects[self.obname]
        self.body.data.use_auto_smooth = False
        self.body.data.shape_keys.animation_data_clear()
        self.armature = bpy.data.objects['Armature']
        self.armature.scale = [100, 100, 100]
        # Load tshirt obj file
        bpy.ops.import_scene.obj(filepath=os.path.join('model', 't-shirt.obj'),
                                 axis_forward='-Z', axis_up='Y')
        self.tshirt = bpy.data.objects['tshirt']

    def deselect(self):
        for o in bpy.data.objects.values():
            o.select_set(False)
        bpy.context.view_layer.objects.active = None

    def bone_name(self, i, bodyname='f_avg'):
        return (self.obname + '_' + self.kintree[i][1])

    # rodrigues transformation. input rotation vector, return rotation matrix
    def rodrigues(self, rotvec):
        # TODO: debug info, in theta = 0 case ,dim of r is (3) otherwise (3,1)
        # TODO: optimize condition case
        theta = np.linalg.norm(rotvec)
        r = (rotvec / theta).reshape(3, 1) if theta > 0. else rotvec
        # print("in rodrigues, r = {0}".format(r.shape))
        cost = np.cos(theta)
        mat = np.array([[0, -r[2], r[1]],
                        [r[2], 0, -r[0]],
                        [-r[1], r[0], 0]], dtype=np.float32)
        return (cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat)

    # Rodrigues to beta shape transformation
    def rodrigues2bshapes(self, pose):
        rod_rots = np.asarray(pose, dtype=np.float32).reshape(-1, 3)
        mat_rots = np.asarray([self.rodrigues(rod_rot) for rod_rot in rod_rots], dtype=np.float32)
        bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel() for mat_rot in mat_rots[1:]])
        return (mat_rots, bshapes)

    # Apply shape and pose to frame in blender
    def apply_shape_pose(self, beta, pose, frame):
        mpose = np.zeros(shape=(self.n_bones, 3, 3), dtype=np.float32)
        pose = pose.reshape(-1, 3)
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
            self.body.data.shape_keys.key_blocks['Shape{0:0>3}'.format(ibeta)].value = val
            self.body.data.shape_keys.key_blocks['Shape{0:0>3}'.format(ibeta)].keyframe_insert('value', index=-1,
                                                                                               frame=frame)

    # Cloth simulate via blender and export obj file
    def simulate_pose(self, beta, poses, export_path):
        tpose = np.zeros(shape=(72), dtype=np.float32)
        # beta =  np.zeros(10, dtype=np.float32)
        self.body.select_set(True)
        bpy.context.view_layer.objects.active = self.body
        frame_end = poses.shape[0]
        # set beta parameter ranges from -10 to 10
        for k in self.body.data.shape_keys.key_blocks.keys():
            self.body.data.shape_keys.key_blocks[k].slider_min = -10
            self.body.data.shape_keys.key_blocks[k].slider_max = 10
            bpy.data.shape_keys['Key'].key_blocks[k].slider_max = 10
            bpy.data.shape_keys['Key'].key_blocks[k].slider_min = -10
        beta_origin = np.array([0, 5, 2, 3, 7, -4, 1, 2, 4, -1], dtype=np.float32)
        self.apply_shape_pose(beta_origin, tpose, frame=0)
        for i, p in enumerate(poses):
            self.apply_shape_pose(beta, p, frame=i + 100)
        # Jump to end point
        bpy.ops.screen.frame_jump(end=False)
        # Set physical properity
        self.deselect()
        avg = bpy.data.objects[self.obname]
        avg.select_set(True)
        bpy.context.view_layer.objects.active = avg
        bpy.ops.object.modifier_add(type='COLLISION')
        self.deselect()

        self.tshirt.select_set(True)  # select tshirt
        bpy.context.view_layer.objects.active = self.tshirt
        bpy.ops.object.modifier_add(type='CLOTH')
        bpy.context.object.modifiers['Cloth'].settings.quality = 10
        bpy.context.object.modifiers['Cloth'].settings.tension_stiffness = 25
        bpy.context.object.modifiers['Cloth'].settings.compression_stiffness = 25
        bpy.context.object.modifiers['Cloth'].settings.shear_stiffness = 5
        bpy.context.object.modifiers['Cloth'].settings.bending_stiffness = 0.5
        bpy.context.object.modifiers['Cloth'].collision_settings.use_self_collision = True
        bpy.context.object.modifiers['Cloth'].collision_settings.collision_quality = 5
        bpy.ops.object.modifier_add(type='COLLISION')
        # Bake
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'  # or 'CPU'
        bpy.context.object.modifiers['Cloth'].point_cache.frame_end = frame_end + 100

        for scene in bpy.data.scenes:
            for object in scene.objects:
                for modifier in object.modifiers:
                    if modifier.type == 'CLOTH':
                        override = {'scene': scene, 'active_object': object, 'point_cache': modifier.point_cache}
                        bpy.ops.ptcache.bake(override, bake=True)
                        break
                        # end bake
        self.deselect()
        # Export frame

        # bpy.context.scene.frame_set(frame_end)
        bpy.data.scenes["Scene"].frame_end = frame_end + 100
        self.tshirt.select_set(True)
        bpy.ops.export_scene.obj(filepath=export_path,
                                 check_existing=False,
                                 use_animation=True, use_materials=False,
                                 use_triangles=True, use_selection=True,
                                 keep_vertex_order=True)

        # free memory
        bpy.ops.ptcache.free_bake_all()
        # Reset to default state
        bpy.ops.wm.read_homefile()


def test():
    ret = global_pose[0]
    return ret


def load_pose(pose_id=0):
    pose_file_path = os.path.join(global_pose_set_dir, global_pose[pose_id] + '.npz')
    # print('npz file path : {0}'.format(pose_file_path))
    pose = np.load(pose_file_path)
    # print(list(pose))
    trans = pose['trans']
    gender = pose['gender']
    mocap_framerate = pose['mocap_framerate']
    betas = pose['betas']
    poses = pose['poses']
    poses[:, 66:72] = 0.0
    return (poses[:, :72])


def load_shape():
    betas = np.load('betas.npz')['betas']
    betas = betas[:, :10]
    return (betas)


if __name__ == '__main__':
    '''
    for pose_id in range(0,15):
        pose = load_pose(pose_id)
        bm = BodyModel()
        export_path = os.path.join(global_pose_set_dir, global_pose[pose_id],'tshirt.obj')
        print('export path = {}'.format(export_path))
        beta =  np.zeros(10, dtype=np.float32)
        bm.simulate_pose(beta, pose, export_path=export_path)
    '''
    betas = load_shape()
    len_instances = len(betas)
    for i in range(len_instances):
        for pose_id in range(0, 15):
            pose = load_pose(pose_id)
            # print('pose {0} = {1}'.format(global_pose[pose_id], pose))
            bm = BodyModel()
            export_path = os.path.join(global_pose_shape_set_dir, 'instance{0:0>3}'.format(i), global_pose[pose_id])
            export_path = os.path.join(export_path, 'tshirt.obj')
            print('export_path = {0}'.format(export_path))
            bm.simulate_pose(betas[i], pose, export_path=export_path)