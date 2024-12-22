#!/usr/bin/env python3

import rospy

import numpy as np
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET

from sensor_msgs.msg import JointState
from hand_landmarker.msg import HandLandmarks

# list of joints in sequential order
URDF_JOINTS = {
    'thumb': [(None, '0f', (1, 2)), ('0f', '0a', (1, 2)), ('0a', '1', (2, 3)), ('1', '2', (3, 4)), ('2', '3', (4, 5))],
    'index': [(None, '0a', (5, 6)), ('0a', '0f', (5, 6)), ('0f', '1', (6, 7)), ('1', '2', (7, 8)), ('2', '3', (8, 9))],
    'mid': [(None, '0a', (9, 10)), ('0a', '0f', (9, 10)), ('0f', '1', (10, 11)), ('1', '2', (11, 12)), ('2', '3', (12, 13))],
    'ring': [(None, '0a', (13, 14)), ('0a', '0f', (13, 14)), ('0f', '1', (14, 15)), ('1', '2', (15, 16)), ('2', '3', (16, 17))],
    'pinky': [(None, '0a', (17, 18)), ('0a', '0f', (17, 18)), ('0f', '1', (18, 19)), ('1', '2', (19, 20)), ('2', '3', (17, 18))]
}
GRASPIT_DOF_BASES = {'index': 0, 'mid': 4, 'pinky': 8, 'ring': 12, 'thumb': 16}

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
def get_joint_name(finger, tup):
    a = tup[0]; b = tup[1]
    a = 'palm' if a is None else f'{finger}{a}'
    b = f'{finger}{b}'
    return f'{a}_{b}'

# https://stackoverflow.com/a/59204638
def find_rotation_matrix_from_vectors(vec1, vec2):
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


class HandJoints:
    def __init__(self):
        self.urdf = ET.parse(rospy.get_param('~urdf')) # parse URDF file
        self.model_vects = np.array([self.xyz_to_array(self.urdf_read_joint(f'palm_{finger}{URDF_JOINTS[finger][0][1]}').find('./origin').attrib['xyz']) for finger in URDF_JOINTS]) # reference vectors to align to later on

        self.joints_pub = rospy.Publisher('joints', JointState, queue_size=10)
        self.lm_sub = rospy.Subscriber('landmarks', HandLandmarks, self.landmarks_cb)

        self.seq = 0

    def urdf_read_joint(self, name):
        return self.urdf.find(f"./joint[@name='{name}']")
    def urdf_read_link(self, name):
        return self.urdf.find(f"./link[@name='{name}']")
    def xyz_to_array(self, s):
        return np.array([float(x) for x in s.split()])

    def landmarks_cb(self, data):
        if len(data.landmarks) == 0: return
        if len(data.landmarks) != 1:
            rospy.logerr('expected only one hand')
            return

        hand_dofs = [0] * 4 * 5
        hand_dof_names = [''] * 4 * 5

        # rotate to align with reference
        joints = np.array([[lm.x, -lm.y, -lm.z] for lm in data.landmarks[0].lms])
        wrist_pos = np.copy(joints[0]) # remember to copy!
        joints -= wrist_pos # centre about wrist
        finger_vectors = joints[[URDF_JOINTS[finger][0][2][0] for finger in URDF_JOINTS]]
        hand_rot, _ = Rotation.align_vectors(self.model_vects, finger_vectors)
        rot_matrix = hand_rot.as_matrix()
        joints = rot_matrix.dot(joints.T).T

        for finger in URDF_JOINTS:
            frame = np.eye(3) # rotation only
            # print(f'Processing finger {finger} (index {n}).')
            for i in range(len(URDF_JOINTS[finger]) - 1):
                joint = URDF_JOINTS[finger][i]
                joint_name = get_joint_name(finger, joint)
                joint_elem = self.urdf_read_joint(joint_name)

                ref_vect = None
                for j in range(i + 1, len(URDF_JOINTS[finger])):
                    joint_end = self.urdf_read_joint(get_joint_name(finger, URDF_JOINTS[finger][j])).find('./origin')
                    if joint_end is None: continue
                    ref_vect = self.xyz_to_array(joint_end.attrib['xyz'])
                    break
                if ref_vect is None:
                    raise RuntimeError('cannot form ref_vect')
                    
                rot_axis = self.xyz_to_array(joint_elem.find('./axis').attrib['xyz']) # rotational axis

                # calculate rotation angle (in joint frame)
                vect = np.linalg.inv(frame).dot(joints[joint[2][1]] - joints[joint[2][0]])
                ref_vect /= np.sqrt(ref_vect.dot(ref_vect)); vect /= np.sqrt(vect.dot(vect))
                ang = calc_rot_angle(ref_vect, vect, rot_axis)

                # print(f' - Joint {i} ({joint[0]} -> {joint[1]}): {np.degrees(ang)} deg')

                # transform joint frame to next one
                rot_align = find_rotation_matrix_from_vectors(np.array([0, 0, 1]), rot_axis) # bring rotation axis to Z axis
                frame_rot = rot_align.dot(Rotation.from_euler('z', ang).as_matrix()).dot(np.linalg.inv(rot_align)) # rotation matrix to align current frame to next frame
                
                frame = frame.dot(frame_rot)
                # frame = frame.dot(frame).dot(make_4x4trans(frame_rot, frame_trans)).dot(np.linalg.inv(frame))

                hand_dofs[GRASPIT_DOF_BASES[finger] + i] = ang.item()
                hand_dof_names[GRASPIT_DOF_BASES[finger] + i] = joint_name
        
        data.header.seq = self.seq; self.seq += 1
        self.joints_pub.publish(JointState(
            header = data.header,
            name = hand_dof_names,
            position = hand_dofs
        ))

def main():
    rospy.init_node('hand_joints', anonymous=True)
    node = HandJoints()

    rospy.loginfo('spinning')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo('shutting down')

if __name__ == '__main__':
    main()