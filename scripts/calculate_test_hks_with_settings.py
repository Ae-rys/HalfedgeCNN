import os
from settings.settings import print_cuda_information, create_settings_string, get_dataset_settings_dict, get_clas_or_seg_settings_dict, \
    get_test_settings_dict, get_general_settings_dict, get_cuda_settings_string


def calculate_test_hks(dataset_name="shrec_16"):
    print_cuda_information()

    dataset_dict           = get_dataset_settings_dict(dataset_name)
    clas_or_seg_settings_dict  = get_clas_or_seg_settings_dict(dataset_dict)
    test_settings_dict     = get_test_settings_dict()
    general_settings_dict  = get_general_settings_dict()

    # dataset settings override test settings, which override general settings
    combined_settings = {**general_settings_dict, **clas_or_seg_settings_dict, **test_settings_dict, **dataset_dict}

    command = 'python calculate_test_hks.py '
    command += '--feat_selection 3 '
    command += create_settings_string(combined_settings)
    command += get_cuda_settings_string()
    print("Command:", command)
    os.system(command)


if __name__ == '__main__':
    """ The script expects the name of the dataset as an argument. A text file conaining the settings for the dataset must exist."""

    calculate_test_hks()