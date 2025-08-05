import os
import numpy as np

# Path to the directory containing .npy files
directory = 'shapenet/shapenet_embedding'

# Get the list of .npy files in the directory
npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]

if npy_files:
    # Read the first .npy file
    first_file_path = os.path.join(directory, npy_files[0])
    data = np.load(first_file_path)
    
    # Print the size of the loaded data
    print(f"First file: {npy_files[0]}")  # 1, 1280, which is the emb dim of laion/CLIP-ViT-bigG-14-laion2B-39B-b160k, what was used by ULIP2 in embedding alignment
    print(f"Shape of the data: {data.shape}")
else:
    print("No .npy files found in the directory.")