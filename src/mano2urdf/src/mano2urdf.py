#!/usr/bin/env python3

import rospkg

import argparse
import glob

import torch
from manopth.manolayer import ManoLayer
from manopth import demo
import open3d as o3d
import numpy as np
import yaml
import os

try:
    PKG_PATH = rospkg.RosPack().get_path('mano2urdf') # our own path
except:
    PKG_PATH = os.path.dirname(os.path.dirname(__file__))

parser = argparse.ArgumentParser(
    prog='mano2urdf',
    description='convert MANO models from DexYCB dataset to URDF descriptions'
)

parser.add_argument('input', help='path to the DexYCB dataset')
parser.add_argument('-o', '--output', default=os.path.dirname(PKG_PATH), help='the directory to create URDF models in')
parser.add_argument('-m', '--mano', default=os.environ.get('MANO_MODELS_PATH', None), help='the directory where MANO model files (particularly MANO_RIGHT.pkl) is stored')

args = parser.parse_args()
INPUT_DIR = args.input
OUTPUT_DIR = args.output
MANO_DIR = args.mano

subjects = [os.path.basename(p) for p in glob.glob(INPUT_DIR + '/calibration/mano*')] # list of subjects to convert

print('reading betas from subjects')
mano_betas = []
for sub in subjects:
    with open(INPUT_DIR + f'/calibration/{sub}/mano.yml') as f:
        mano_betas.append(yaml.safe_load(f)['betas'])
mano_betas = torch.from_numpy(np.array(mano_betas, dtype=np.float32)) # convert to pytorch (for ManoLayer)

mano_layer = ManoLayer(mano_root=MANO_DIR, use_pca=True, ncomps=45, side='right')

mano_faces = mano_layer.th_faces.detach().cpu().numpy()
segment_faces = np.load(PKG_PATH + '/sealed_faces.npy', allow_pickle=True).item()['sealed_faces_color_right'][:1538] # MANO model has 1538 faces; they are also the first 1538 faces in sealed_faces.npy (the rest are to close the model off)
segment_faces = { id: np.where(segment_faces == id) for id in np.unique(segment_faces) }

mano_thetas = torch.zeros(len(subjects), 45 + 3)
all_verts, all_joints = mano_layer(mano_thetas, mano_betas) # calculate vertices and joints
all_verts = (all_verts / 1000).detach().cpu().numpy() # convert back to numpy for processing, and also convert milimetres to metres
all_joints = (all_joints / 1000).detach().cpu().numpy()

finger_bases = {'thumb': 1, 'index': 5, 'mid': 9, 'ring': 13, 'pinky': 17} # base joint indices for fingers
seg_names = ['ring1', 'index2', 'pinky1', 'mid1', 'mid3', 'ring3', 'pinky3', 'thumb1', 'palm', 'thumb2', 'index1', 'index3', 'thumb3', 'pinky2', 'mid2', 'ring2'] # names of segments
seg_map = dict(enumerate([13, 6, 17, 9, 11, 15, 19, 1, 0, 2, 5, 7, 3, 18, 10, 14])) # mappings from segments to their root joint indices

print('reading templates')
templates = dict()
for fn in glob.glob(PKG_PATH + '/pkg_template/*.template'):
    with open(fn, 'r') as f:
        templates[os.path.basename(fn).replace('.template', '')] = f.read()

for (nsub, sub) in enumerate(subjects):
    MODEL_DIR = OUTPUT_DIR + '/' + sub
    print(f'generating URDF model for {sub} - the model will be saved at {MODEL_DIR}')

    STL_DIR = MODEL_DIR + '/models'
    os.makedirs(STL_DIR, exist_ok=True)

    verts = all_verts[nsub]; joints = all_joints[nsub]

    # centre about wrist
    wrist_pos = joints[0]; verts -= wrist_pos; joints -= wrist_pos

    # break down model into segments and save them
    print(f' - saving 3D models')
    for segid in segment_faces:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(mano_faces[segment_faces[segid]])
        mesh.remove_unreferenced_vertices() # clean up vertices
        mesh.compute_vertex_normals()
        mesh.translate(-joints[seg_map[segid]]) # translate back to origin
        o3d.io.write_triangle_mesh(STL_DIR + f'/{seg_names[segid]}.stl', mesh) # save model as STL file

    # write URDF file
    print(f' - saving URDF file')
    MODELS_PATH = f'package://{sub}/models'
    with open(MODEL_DIR + f'/{sub}.urdf', 'w') as f:
        f.write(f'<?xml version="1.0"?>\n<robot name="{sub}">\n')
        f.write(f'<link name="palm"><visual><geometry><mesh filename="{MODELS_PATH}/palm.stl"/></geometry></visual></link>\n')

        for finger in finger_bases:
            prev_origin = joints[0] # previous origin - we start from wrist (which would be zero anyway)
            prev_joint = 'palm'

            for i in range(4):
                joint = f'{finger}{i}' # joint name
                origin = joints[finger_bases[finger] + i]; origin_offset = origin - prev_origin

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
                        cmc_mcp = joints[finger_bases[finger] + i + 1] - origin # CMC-MCP vector
                        rot_axis = np.cross(cmc_mcp, np.array([0, -1, 0])); rot_axis /= np.sqrt(rot_axis.dot(rot_axis)) # rotation axis
                        f.write(f'<joint name="{prev_joint}_{joint}" type="revolute">\n<limit effort="1000.0" velocity="0.5" lower="-1.57" upper="1.57"/>\n<parent link="{prev_joint}"/>\n<child link="{joint}"/>\n')
                        f.write(f'<origin xyz="' + ' '.join(f'{x:.18f}' for x in origin_offset.tolist()) + '"/>\n')
                        f.write(f'<axis xyz="' + ' '.join(f'{x:.18f}' for x in rot_axis.tolist()) + '"/></joint>\n')

                prev_joint = joint
                prev_origin = origin

        f.write('</robot>\n')

        # write other files
        print(' - saving templated files')
        for fn in templates:
            with open(MODEL_DIR + '/' + fn, 'w') as f:
                f.write(templates[fn].replace('{SUBJECT}', sub))

# print instructions
print('-' * 70)
print('WHAT\'S NEXT: Follow the steps below on a ROS Melodic installation')
print('with urdf2graspit to convert the generated URDF descriptions to those')
print('used by GraspIt:')
print('  1. Copy the mano_* directories in src/ to the installation\'s')
print('     ROS workspace\'s src/ directory.')
print('  2. Set up the workspace, and run roscore in a separate terminal.')
print('  3. Run the following commands:')
for sub in subjects:
    print(f'rosrun urdf2graspit urdf2graspit_node `rospack find {sub}`/{sub}.urdf $GRASPIT palm palm_index0a palm_mid0a palm_pinky0a palm_ring0a palm_thumb0f')
print('  4. Copy the resulting mano_* directories in $GRASPIT/models/robots')
print('     back to your installation.')
