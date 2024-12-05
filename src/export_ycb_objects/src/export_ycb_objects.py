#!/usr/bin/env python3

import open3d as o3d
import cv2
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    prog='export_ycb_objects',
    description='export YCB objects in DexYCB dataset to GraspIt format'
)

parser.add_argument('input', help='path to the DexYCB dataset')
parser.add_argument('-o', '--output', default=os.environ.get('GRASPIT'), help='the GraspIt data directory to output files to')

args = parser.parse_args()
INPUT_DIR = args.input
OUTPUT_DIR = args.output + '/models/objects'

_YCB_CLASSES = {
     1: '002_master_chef_can',
     2: '003_cracker_box',
     3: '004_sugar_box',
     4: '005_tomato_soup_can',
     5: '006_mustard_bottle',
     6: '007_tuna_fish_can',
     7: '008_pudding_box',
     8: '009_gelatin_box',
     9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}

def write_vrml(name: str, mesh: o3d.geometry.TriangleMesh, color = None):
    with open(name, 'w') as f:
        f.write('#VRML V2.0 utf8\n\n') # initial lines

        f.write('Shape {\n')
        
        if color is not None:
            f.write('\tappearance Appearance {\n\t\tmaterial Material {\n\t\t\tdiffuseColor ' + ' '.join(str(x) for x in color) + '\n\t\t}\n\t}\n')

        f.write('\tgeometry IndexedFaceSet {\n')

        # coord Coordinate
        f.write('\t\tcoord Coordinate {\n\t\t\tpoint [\n')
        for point in np.asarray(mesh.vertices).tolist():
            f.write('\t\t\t\t' + ' '.join(str(x) for x in point) + '\n')
        f.write('\t\t\t]\n\t\t}\n\t\tcoordIndex [\n')
        for triangle in np.asarray(mesh.triangles).tolist():
            f.write('\t\t\t' + ' '.join(str(x) for x in triangle) + ' -1\n')
        f.write('\t\t]\n\t\tnormal Normal {\n\t\t\tvector [\n')
        for vect in np.asarray(mesh.vertex_normals).tolist():
            f.write('\t\t\t\t' + ' '.join(str(x) for x in vect) + '\n')
        f.write('\t\t\t]\n\t\t}\n')

        f.write('\t}\n}\n')

for i in _YCB_CLASSES:
    object_name = _YCB_CLASSES[i]
    print(f'Converting object {i} ({object_name}) to GraspIt format.')
    
    MODEL_DIR = f'{INPUT_DIR}/models/{object_name}'
    mesh = o3d.io.read_triangle_mesh(MODEL_DIR + '/textured_simple.obj')
    mesh.translate(-mesh.get_center())
    mesh.scale(1000, [0, 0, 0])
    print(f' - Input mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris')

    # simplify mesh
    voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 32
    # print(f'voxel_size = {voxel_size:e}')
    mesh = mesh.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average
    )
    print(f' - Simplified mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris')

    mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh_simp])

    # calculate average colour from texture map
    texture_map = cv2.imread(MODEL_DIR + '/texture_map.png'); texture_map = cv2.cvtColor(texture_map, cv2.COLOR_BGR2RGB)
    pixels = texture_map.reshape(-1, 3)
    avg_colour = np.mean(pixels[np.any(pixels > 0, axis=1)], axis=0) / 255    

    MODEL_NAME = f'dexycb_{i}'
    write_vrml(f'{OUTPUT_DIR}/{MODEL_NAME}.wrl', mesh, avg_colour)
    with open(f'{OUTPUT_DIR}/{MODEL_NAME}.xml', 'w') as f:
        f.write('\n'.join([
            '<?xml version="1.0" ?>',
            '<root>',
            '\t<material>plastic</material>',
            '\t<mass>300</mass>',
            f'\t<geometryFile type="Inventor">{MODEL_NAME}.wrl</geometryFile>',
            '</root>'
        ]))

print(f'Object files have been saved to {OUTPUT_DIR}')