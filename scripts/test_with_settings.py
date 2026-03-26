import os
import numpy as np
from train_util.train_util import get_dataset_name_from_command_line
from settings.settings import print_cuda_information, get_clas_or_seg_settings_dict, create_settings_string, get_cuda_settings_string, \
    get_dataset_settings_dict, get_test_settings_dict, get_general_settings_dict


def test(dataset_name, model=None, export_folder=None, export_hks_values=False):
    print_cuda_information()

    dataset_dict           = get_dataset_settings_dict(dataset_name)
    clas_or_seg_settings_dict  = get_clas_or_seg_settings_dict(dataset_dict)
    test_settings_dict     = get_test_settings_dict()
    general_settings_dict  = get_general_settings_dict()

    # dataset settings override test settings, which override general settings
    combined_settings = {**general_settings_dict, **clas_or_seg_settings_dict, **test_settings_dict, **dataset_dict}


    # -W ignore used because of VisibleDeprecationWarning that otherwise clutter output
    command = 'python -W ignore test.py '
    command += create_settings_string(combined_settings)
    command += get_cuda_settings_string()
    if export_hks_values:
        command += '--export_hks_values '
    if model:
        command += '--model ' + model + ' '
    if export_folder is not None:
        command += '--export_folder '+export_folder+ ' '
    print("Command:", command)
    os.system(command)


if __name__ == '__main__':
    """ The script expects the name of the dataset as an argument. A text file conaining the settings for the dataset must exist."""

    import argparse
    parser = argparse.ArgumentParser("test with settings")
    parser.add_argument('--model', default='best_model.pth', type=str,
                        help="give the name of the model to be tested, e.g. 'best_model.pth'")
    parser.add_argument('--dataset', default='shrec_16', type=str,
                        help="give the name of the dataset to be tested, e.g. 'shrec_16'")
    parser.add_argument('--export_folder', default='export/classification', type=str,
                        help="path to the folder where HKS values will be exported (if --export_hks_values is also given)")
    parser.add_argument('--export_hks_values', action='store_true',
                        help="flag to export HKS values for the meshes in the test set (if --export_folder is also given)")
    args = parser.parse_args()
    
    test(dataset_name=args.dataset, model=args.model, export_folder=args.export_folder, export_hks_values=args.export_hks_values)