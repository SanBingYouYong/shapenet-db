import numpy as np
from plyfile import PlyData, PlyElement

def save_as_ply(npy_file, ply_file):
    # Load the .npy file
    points = np.load(npy_file)

    # Ensure the points are in the correct format
    vertices = [(point[0], point[1], point[2]) for point in points]
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

    # Create a PlyElement
    vertex_array = np.array(vertices, dtype=vertex_dtype)
    ply_element = PlyElement.describe(vertex_array, 'vertex')

    # Write to .ply file
    PlyData([ply_element]).write(ply_file)

# File paths
npy_file = 'found_chair.npy'
ply_file = 'found_chair.ply'

# Convert and save
save_as_ply(npy_file, ply_file)