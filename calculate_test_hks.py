# Uses hks functions to calculate hks values for meshes in the test set and exports them as .npy files

import os
import numpy as np
from models.layers.hks import compute_hks_features
from models.layers.half_edge_mesh_prepare import from_scratch
from scripts.settings.settings import get_dataset_settings_dict, create_settings_string
from options.test_options import TestOptions

def calculate_and_export_hks(mesh_path, export_folder, k=100):
    
    opt = TestOptions().parse()
    mesh_data = from_scratch(file=mesh_path, opt=opt)
    hks_features = compute_hks_features(opt.t, k, mesh_data, None)
    
    # Create export folder if it doesn't exist
    os.makedirs(export_folder, exist_ok=True)
    # Save HKS features as .npy file
    export_path = os.path.join(export_folder, os.path.basename(mesh_path).replace('.obj', '_initial_hks.npy'))
    np.save(export_path, hks_features)
    print(f"HKS features for {mesh_path} saved to {export_path}")


if __name__ == '__main__':
    test_mesh_folder = 'datasets/shrec_16'
    export_folder = 'checkpoints/shrec_16/export/classification'
    
    # For each folder in test_mesh_folder, look for the test folder and calculate HKS for meshes in that folder
    for folder in os.listdir(test_mesh_folder):
        folder_path = os.path.join(test_mesh_folder, folder)
        if os.path.isdir(folder_path):
            test_folder_path = os.path.join(folder_path, 'test')
            if os.path.isdir(test_folder_path):
                for mesh_file in os.listdir(test_folder_path):
                    if mesh_file.endswith('.obj'):
                        mesh_path = os.path.join(test_folder_path, mesh_file)
                        calculate_and_export_hks(mesh_path, export_folder)
