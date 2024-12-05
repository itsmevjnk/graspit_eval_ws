#!/usr/bin/env python3

import rospy
import rospkg

from scipy.spatial.transform import Rotation
import torch
from manopth.manolayer import ManoLayer
from manopth import demo
import numpy as np
import yaml
import os
import xml.etree.ElementTree as ET

from graspit_interface.srv import *
from geometry_msgs.msg import Pose, Point, Quaternion


# mapping from subject number to dataset directory and MANO model name
DEXYCB_SUBJECT_DIRS = {
    1: '20200709-subject-01',
    2: '20200813-subject-02',
    3: '20200820-subject-03',
    4: '20200903-subject-04',
    5: '20200908-subject-05',
    6: '20200918-subject-06',
    7: '20200928-subject-07',
    8: '20201002-subject-08',
    9: '20201015-subject-09',
    10: '20201022-subject-10'
}
DEXYCB_SUBJECT_MODELS = {
    1: 'mano_20200709_140042_subject-01_right',
    2: 'mano_20200813_143449_subject-02_right',
    3: 'mano_20200820_133405_subject-03_right',
    4: 'mano_20200903_101911_subject-04_right',
    5: 'mano_20200908_140650_subject-05_right',
    6: 'mano_20200918_110920_subject-06_right',
    7: 'mano_20200807_132210_subject-07_right',
    8: 'mano_20201002_103251_subject-08_right',
    9: 'mano_20200514_142106_subject-09_right',
    10: 'mano_20201022_105224_subject-10_right'
}

# list of joints in sequential order
URDF_JOINTS = {
    'thumb': [(None, '0f', (1, 2)), ('0f', '0a', (1, 2)), ('0a', '1', (2, 3)), ('1', '2', (3, 4)), ('2', '3', (4, 5))],
    'index': [(None, '0a', (5, 6)), ('0a', '0f', (5, 6)), ('0f', '1', (6, 7)), ('1', '2', (7, 8)), ('2', '3', (8, 9))],
    'mid': [(None, '0a', (9, 10)), ('0a', '0f', (9, 10)), ('0f', '1', (10, 11)), ('1', '2', (11, 12)), ('2', '3', (12, 13))],
    'ring': [(None, '0a', (13, 14)), ('0a', '0f', (13, 14)), ('0f', '1', (14, 15)), ('1', '2', (15, 16)), ('2', '3', (16, 17))],
    'pinky': [(None, '0a', (17, 18)), ('0a', '0f', (17, 18)), ('0f', '1', (18, 19)), ('1', '2', (19, 20)), ('2', '3', (17, 18))]
}

graspit_dof_bases = {'index': 0, 'mid': 4, 'pinky': 8, 'ring': 12, 'thumb': 16} # DOF index bases

# TODO
DEXYCB_PATH = os.environ.get('DEX_YCB_DIR')
MANO_DIR = os.environ.get('MANO_MODELS_PATH')

# TODO
DEXYCB_SUBJECT = 1
DEXYCB_SEQUENCE = '20200709_141754'

rospy.init_node('scene_reconstructor')

mano_layer = ManoLayer(mano_root=MANO_DIR, use_pca=True, ncomps=45, side='right', flat_hand_mean=False)

# read MANO betas (per subject)
DEXYCB_MODEL = DEXYCB_SUBJECT_MODELS[DEXYCB_SUBJECT]
with open(f'{DEXYCB_PATH}/calibration/{DEXYCB_MODEL}/mano.yml') as f: mano_betas = yaml.safe_load(f)['betas']
mano_betas = torch.from_numpy(np.array(mano_betas, dtype=np.float32))

# read URDF file (per subject)
urdf = ET.parse(rospkg.RosPack().get_path(DEXYCB_MODEL) + f'/{DEXYCB_MODEL}.urdf').getroot()
# helpers
def read_joint(name):
    return urdf.find(f"./joint[@name='{name}']")
def read_link(name):
    return urdf.find(f"./link[@name='{name}']")
def xyz_to_array(s):
    return np.array([float(x) for x in s.split()])
model_finger_vectors = np.array([xyz_to_array(read_joint(f'palm_{finger}{URDF_JOINTS[finger][0][1]}').find('./origin').attrib['xyz']) for finger in URDF_JOINTS]) # reference vectors to align to later on

# read sequence metadata (per sequence)
SEQUENCE_DIR = f'{DEXYCB_PATH}/{DEXYCB_SUBJECT_DIRS[DEXYCB_SUBJECT]}/{DEXYCB_SEQUENCE}'
with open(f'{SEQUENCE_DIR}/meta.yml', 'r') as f:
    seq_meta = yaml.safe_load(f)
    seq_grasp_idx = seq_meta['ycb_grasp_ind']; seq_grasp_obj = seq_meta['ycb_ids'][seq_grasp_idx] # grasped object index and DexYCB ID
    seq_frames = seq_meta['num_frames']

# read sequence's pose data
seq_pose = np.load(f'{SEQUENCE_DIR}/pose.npz')
seq_hand_thetas = seq_pose['pose_m'][:,0,:48]; seq_hand_trans = seq_pose['pose_m'][:,0,48:]
seq_obj_trans = seq_pose['pose_y'][:,seq_grasp_idx,-3:]
seq_obj_orient = seq_pose['pose_y'][:,seq_grasp_idx,:-3]
if seq_obj_orient.shape[-1] == 3: # rotation vector - convert to quaternions before proceeding
    seq_obj_orient = Rotation.from_rotvec(seq_obj_orient).as_quat()

# generate MANO hands
seq_hand_betas = torch.tile(mano_betas, (seq_frames, 1)) # we're using the same betas for all frames
all_verts, all_joints = mano_layer(torch.from_numpy(seq_hand_thetas), seq_hand_betas, torch.from_numpy(seq_hand_trans)) # process all frames in one operation

all_joints = all_joints.detach().cpu().numpy() / 1000 # convert to numpy (and also convert to metres)

# calculate angle from a to b, given the rotational axis
def calc_rot_angle(a, b, rot):
    # project a and b onto plane with rot as normal
    a -= a.dot(rot) * rot
    b -= b.dot(rot) * rot

    # calculate rotation angle
    ang = np.arccos(a.dot(b) / np.sqrt(a.dot(a) * b.dot(b)))
    if np.cross(a, b).dot(rot) < 0: ang *= -1 # consider rotation direction

    return ang

# generate joint name (in our URDF description's naming convention)
def joint_name(finger, tup):
    a = tup[0]; b = tup[1]
    a = 'palm' if a is None else f'{finger}{a}'
    b = f'{finger}{b}'
    return f'{a}_{b}'

# https://stackoverflow.com/a/59204638
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    if np.all(np.isclose(kmat, np.zeros((3, 3)))): return np.eye(3)

    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

rospy.ServiceProxy('/graspit/setCheckCollision', SetCheckCollision)(False)

clear_world = rospy.ServiceProxy('/graspit/clearWorld', ClearWorld)
import_robot = rospy.ServiceProxy('/graspit/importRobot', ImportRobot)
import_object = rospy.ServiceProxy('/graspit/importGraspableBody', ImportGraspableBody)

def ros_point(trans):
    if trans is np.ndarray: trans = trans.tolist()
    return Point(trans[0], trans[1], trans[2])
def ros_quaternion(quat):
    if quat is np.ndarray: quat = quat.tolist()
    return Quaternion(quat[0], quat[1], quat[2], quat[3])
def ros_pose(trans, rot):
    return Pose(ros_point(trans), ros_quaternion(rot))

# command GraspIt to reset world
clear_world()
import_robot(DEXYCB_MODEL, Pose(ros_point(seq_hand_trans[0]), Quaternion(0, 0, 0, 1)))
import_object(f'dexycb_{seq_grasp_obj}', ros_pose(seq_obj_trans[0], seq_obj_orient[0]))

set_robot_dof = rospy.ServiceProxy('/graspit/forceRobotDof', ForceRobotDOF)
set_object_pose = rospy.ServiceProxy('/graspit/setGraspableBodyPose', SetGraspableBodyPose)
set_robot_pose = rospy.ServiceProxy('/graspit/setRobotPose', SetRobotPose)
compute_quality = rospy.ServiceProxy('/graspit/computeQuality', ComputeQuality)
for nframe in range(seq_frames): # process each frame
    rospy.loginfo(f'processing frame {nframe}')
    hand_dofs = [0] * 4 * 5

    # rotate to align with reference
    joints = all_joints[nframe]
    wrist_pos = np.copy(joints[0]) # remember to copy!
    joints -= wrist_pos # centre about wrist
    finger_vectors = joints[[URDF_JOINTS[finger][0][2][0] for finger in URDF_JOINTS]]
    hand_rot, rssd = Rotation.align_vectors(model_finger_vectors, finger_vectors)
    rot_matrix = hand_rot.as_matrix()
    joints = rot_matrix.dot(joints.T).T

    for finger in URDF_JOINTS:
        frame = np.eye(3) # rotation only
        # print(f'Processing finger {finger} (index {n}).')
        for i in range(len(URDF_JOINTS[finger]) - 1):
            joint = URDF_JOINTS[finger][i]
            joint_elem = read_joint(joint_name(finger, joint))

            ref_vect = None
            for j in range(i + 1, len(URDF_JOINTS[finger])):
                joint_end = read_joint(joint_name(finger, URDF_JOINTS[finger][j])).find('./origin')
                if joint_end is None: continue
                ref_vect = xyz_to_array(joint_end.attrib['xyz'])
                break
            if ref_vect is None:
                raise RuntimeError('cannot form ref_vect')
                
            rot_axis = xyz_to_array(joint_elem.find('./axis').attrib['xyz']) # rotational axis

            # calculate rotation angle (in joint frame)
            vect = np.linalg.inv(frame).dot(joints[joint[2][1]] - joints[joint[2][0]])
            ref_vect /= np.sqrt(ref_vect.dot(ref_vect)); vect /= np.sqrt(vect.dot(vect))
            ang = calc_rot_angle(ref_vect, vect, rot_axis)

            # print(f' - Joint {i} ({joint[0]} -> {joint[1]}): {np.degrees(ang)} deg')

            # transform joint frame to next one
            rot_align = rotation_matrix_from_vectors(np.array([0, 0, 1]), rot_axis) # bring rotation axis to Z axis
            frame_rot = rot_align.dot(Rotation.from_euler('z', ang).as_matrix()).dot(np.linalg.inv(rot_align)) # rotation matrix to align current frame to next frame
            
            frame = frame.dot(frame_rot)
            # frame = frame.dot(frame).dot(make_4x4trans(frame_rot, frame_trans)).dot(np.linalg.inv(frame))

            hand_dofs[graspit_dof_bases[finger] + i] = ang.item()

    hand_trans = wrist_pos.tolist()
    hand_orient = hand_rot.inv().as_quat().tolist()

    obj_trans = seq_obj_trans[nframe].tolist()
    obj_orient = seq_obj_orient[nframe].tolist()

    set_robot_dof(0, hand_dofs); set_robot_pose(0, ros_pose(hand_trans, hand_orient))
    set_object_pose(0, ros_pose(obj_trans, obj_orient))
    # rospy.loginfo(f'set dof ret={ret}')

    quality_resp = compute_quality(0)
    result = quality_resp.result
    volume = quality_resp.volume; epsilon = quality_resp.epsilon
    rospy.loginfo(f'ret={result} v={volume} e={epsilon}')
    