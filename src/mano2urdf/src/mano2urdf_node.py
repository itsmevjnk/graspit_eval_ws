#!/usr/bin/env python3

import rospy
import rospkg

import torch
from manopth.manolayer import ManoLayer
import open3d as o3d
import numpy as np
import os
import glob
import time
import sys

from hand_calibration.msg import HandShape

try:
    PKG_PATH = rospkg.RosPack().get_path('mano2urdf') # our own path
except:
    PKG_PATH = os.path.dirname(os.path.dirname(__file__))

FINGER_BASES = {'thumb': 1, 'index': 5, 'mid': 9, 'ring': 13, 'pinky': 17} # base joint indices for fingers
SEG_NAMES = ['ring1', 'index2', 'pinky1', 'mid1', 'mid3', 'ring3', 'pinky3', 'thumb1', 'palm', 'thumb2', 'index1', 'index3', 'thumb3', 'pinky2', 'mid2', 'ring2'] # names of segments
SEG_MAP = dict(enumerate([13, 6, 17, 9, 11, 15, 19, 1, 0, 2, 5, 7, 3, 18, 10, 14])) # mappings from segments to their root joint indices

TEMPLATES = dict()
for fn in glob.glob(PKG_PATH + '/pkg_template/*.template'):
    with open(fn, 'r') as f:
        TEMPLATES[os.path.basename(fn).replace('.template', '')] = f.read()

SEGMENT_FACES = np.load(PKG_PATH + '/sealed_faces.npy', allow_pickle=True).item()['sealed_faces_color_right'][:1538] # MANO model has 1538 faces; they are also the first 1538 faces in sealed_faces.npy (the rest are to close the model off)
SEGMENT_FACES = {
    id: np.where(SEGMENT_FACES == id) for id in np.unique(SEGMENT_FACES)
}


class Mano2URDF:
    def __init__(self):
        self.sub = rospy.Subscriber('shape', HandShape, self.shape_cb) # our node will only process one message and exit

        self.mano = ManoLayer(mano_root=rospy.get_param('~mano', os.environ.get('MANO_MODELS_PATH', None)), use_pca=True, ncomps=45, side='right')
        self.output = rospy.get_param('~output', os.path.dirname(PKG_PATH))
        self.mano_faces = self.mano.th_faces.detach().cpu().numpy()

        self.done = False

    def shape_cb(self, data):
        if self.done: return

        if not data.right:
            rospy.logwarn('ignoring detection: only right hand is supported')
            return

        betas = torch.FloatTensor([data.betas])
        thetas = torch.zeros(1, 45 + 3)

        verts, joints = self.mano(thetas, betas) # calculate vertices and joints
        verts = (data.scale * verts[0] / 1000).detach().cpu().numpy() # convert back to numpy for processing, and also convert milimetres to metres
        joints = (joints[0] / 1000).detach().cpu().numpy()

        name = time.strftime('mano_%Y%m%d_%H%M%S')
        MODEL_DIR = self.output + '/' + name
        rospy.loginfo(f'saving URDF model to {MODEL_DIR}')

        STL_DIR = MODEL_DIR + '/models'
        os.makedirs(STL_DIR, exist_ok=True)

        rospy.loginfo(f'saving 3D models')
        for segid in SEGMENT_FACES:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(self.mano_faces[SEGMENT_FACES[segid]])
            mesh.remove_unreferenced_vertices() # clean up vertices
            mesh.compute_vertex_normals()
            mesh.translate(-joints[SEG_MAP[segid]]) # translate back to origin
            o3d.io.write_triangle_mesh(STL_DIR + f'/{SEG_NAMES[segid]}.stl', mesh) # save model as STL file

        # write URDF file
        rospy.loginfo(f'saving URDF file')
        MODELS_PATH = f'package://{name}/models'
        with open(MODEL_DIR + f'/{name}.urdf', 'w') as f:
            f.write(f'<?xml version="1.0"?>\n<robot name="{name}">\n')
            f.write(f'<link name="palm"><visual><geometry><mesh filename="{MODELS_PATH}/palm.stl"/></geometry></visual></link>\n')

            for finger in FINGER_BASES:
                prev_origin = joints[0] # previous origin - we start from wrist (which would be zero anyway)
                prev_joint = 'palm'

                for i in range(4):
                    joint = f'{finger}{i}' # joint name
                    origin = joints[FINGER_BASES[finger] + i]; origin_offset = origin - prev_origin

                    is_thumb = finger == 'thumb'
                    if i == 0: # first joint - we actually have 2 joints, a(dduction/bduction) and f(lex), and also the fixed base
                        if is_thumb: # thumb would have flex before adduction
                            joint = f'{finger}{i}f' # flex
                            f.write(f'<joint name="{prev_joint}_{joint}" type="revolute">\n<limit effort="1000.0" velocity="0.5" lower="-1.57" upper="1.57"/>\n<parent link="{prev_joint}"/>\n<child link="{joint}"/>\n<axis xyz="-1 0 0"/>\n')
                        else:
                            joint = f'{finger}{i}a' # abduction
                            f.write(f'<joint name="{prev_joint}_{joint}" type="revolute">\n<limit effort="1000.0" velocity="0.5" lower="-1.57" upper="1.57"/>\n<parent link="{prev_joint}"/>\n<child link="{joint}"/>\n<axis xyz="0 -1 0"/>\n')

                        f.write(f'<origin xyz="' + ' '.join(f'{x:.18f}' for x in origin_offset.tolist()) + '"/></joint>\n')
                        f.write(f'<link name="{joint}"/>\n')
                        prev_joint = joint

                        if is_thumb:
                            joint = f'{finger}{i}a'
                            f.write(f'<joint name="{prev_joint}_{joint}" type="revolute">\n<limit effort="1000.0" velocity="0.5" lower="-1.57" upper="1.57"/>\n<parent link="{prev_joint}"/>\n<child link="{joint}"/>\n<axis xyz="0 -1 0"/></joint>\n')
                        else:
                            joint = f'{finger}{i}f'
                            f.write(f'<joint name="{prev_joint}_{joint}" type="revolute">\n<limit effort="1000.0" velocity="0.5" lower="-1.57" upper="1.57"/>\n<parent link="{prev_joint}"/>\n<child link="{joint}"/><axis xyz="0 0 1"/></joint>\n')
                    
                        f.write(f'<link name="{joint}"><visual><geometry><mesh filename="{MODELS_PATH}/{finger}1.stl"/></geometry></visual></link>\n')
                    elif i == 3: # last joint (tip)
                        f.write(f'<link name="{joint}"/>\n')
                        f.write(f'<joint name="{prev_joint}_{joint}" type="fixed">\n<parent link="{prev_joint}"/>\n<child link="{joint}"/>\n')
                        f.write(f'<origin xyz="' + ' '.join(f'{x:.18f}' for x in origin_offset.tolist()) + '"/>\n')
                        f.write('</joint>\n')
                    else: # next joints
                        f.write(f'<link name="{joint}"><visual><geometry><mesh filename="{MODELS_PATH}/{finger}{i+1}.stl"/></geometry></visual></link>\n')
                        if finger != 'thumb': # next joints are simpler
                            f.write(f'<joint name="{prev_joint}_{joint}" type="revolute">\n<limit effort="1000.0" velocity="0.5" lower="-1.57" upper="1.57"/>\n<parent link="{prev_joint}"/>\n<child link="{joint}"/>\n')
                            f.write(f'<origin xyz="' + ' '.join(f'{x:.18f}' for x in origin_offset.tolist()) + '"/>\n')
                            f.write(f'<axis xyz="0 0 1"/></joint>\n')
                        else: # CMC-MCP / MCP-IP
                            cmc_mcp = joints[FINGER_BASES[finger] + i + 1] - origin # CMC-MCP vector
                            rot_axis = np.cross(cmc_mcp, np.array([0, -1, 0])); rot_axis /= np.sqrt(rot_axis.dot(rot_axis)) # rotation axis
                            f.write(f'<joint name="{prev_joint}_{joint}" type="revolute">\n<limit effort="1000.0" velocity="0.5" lower="-1.57" upper="1.57"/>\n<parent link="{prev_joint}"/>\n<child link="{joint}"/>\n')
                            f.write(f'<origin xyz="' + ' '.join(f'{x:.18f}' for x in origin_offset.tolist()) + '"/>\n')
                            f.write(f'<axis xyz="' + ' '.join(f'{x:.18f}' for x in rot_axis.tolist()) + '"/></joint>\n')

                    prev_joint = joint
                    prev_origin = origin

            f.write('</robot>\n')

        # write other files
        rospy.loginfo('saving templated files')
        for fn in TEMPLATES:
            with open(MODEL_DIR + '/' + fn, 'w') as f:
                f.write(TEMPLATES[fn].replace('{SUBJECT}', name))

        rospy.loginfo('model has been saved, exiting')
        self.done = True; rospy.signal_shutdown('model saved')

def main():
    rospy.init_node('mano2urdf', anonymous=True)
    node = Mano2URDF()

    rospy.loginfo('spinning')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo('shutting down')

if __name__ == '__main__':
    main()