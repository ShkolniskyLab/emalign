from pathlib import Path
import sys
import subprocess
import argparse
import os

DEFAULT_DOWNSAMPLE = 64
DEFAULT_N_PROJS = 30


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-v1', '--vol1', help='Full path of first mrc input file', type=check_mrc_file_exist)
    parser.add_argument('-v2', '--vol2', help='Full path of second mrc input file', type=check_mrc_file_exist)
    parser.add_argument('-o', '--output-vol', help='Full path of output mrc file', type=check_mrc_file_not_exist)
    parser.add_argument('--downsample', help='Dimension to downsample input volumes', default=DEFAULT_DOWNSAMPLE,
                        type=check_positive_int)
    parser.add_argument('--n-projs', help='Number of projections', default=DEFAULT_N_PROJS,
                        type=check_positive_int)
    parser.add_argument('--no-refine', help='Skip optimization to refine aligment parameters', action='store_true', default=False)
    parser.add_argument('--output-parameters', help='Full path of output txt file for alignment parameters', default=None,
                        type=check_txt_file_not_exist)
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print progress messages', default=False)

    # Make emalign to generate two volumes, which can be used to test the 
    # alignment algorithm.
    parser.add_argument('--make-test-data', help='Generate test volumes to test emalign', action='store_true',
                        default=False)

    # Version
    parser.add_argument('--version', help='print program version and exit', action='store_true', default=False)

    args = parser.parse_args()
    return args


def check_positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def check_mrc_file_exist(output_file):
    if not os.path.isfile(output_file):
        raise argparse.ArgumentTypeError("There is no file with the name %s." % output_file)
    if output_file[-4:] != '.mrc':
        raise argparse.ArgumentTypeError('%s is not an mrc file.' % output_file)
    return output_file


def check_mrc_file_not_exist(output_file):
    if os.path.isfile(output_file):
        raise argparse.ArgumentTypeError("There is already a file with the name %s." % output_file)
    output_dir, _ = os.path.split(output_file)
    if output_dir == "":    # No directory not specified. Use current directory.
        output_dir = "."
        output_file = os.path.join(output_dir, output_file)
    if not os.path.isdir(output_dir):
        raise argparse.ArgumentTypeError(
            'Directory %s does not exist. Please specify a file in existing directory.' % output_dir)
    if output_file[-4:] != '.mrc':
        raise argparse.ArgumentTypeError('%s is not an mrc file.' % output_file)
    return output_file


def check_txt_file_not_exist(output_file):
    if os.path.isfile(output_file):
        raise argparse.ArgumentTypeError("There is already a file with the name %s." % output_file)
    output_dir, _ = os.path.split(output_file)
    if output_dir == "":    # No directory not specified. Use current directory.
        output_dir = "."
        output_file = os.path.join(output_dir, output_file)
    if not os.path.isdir(output_dir):
        raise argparse.ArgumentTypeError(
            'Directory %s does not exist. Please specify a file in existing directory.' % output_dir)
    if output_file[-4:] != '.txt':
        raise argparse.ArgumentTypeError('%s is not an txt file.' % output_file)
    return output_file


def get_args():
    # print("\n")
    while True:
        vol1 = input('Enter full path of reference volume MRC file: ')
        if vol1[-4:] != '.mrc':
            print('Please enter an MRC file')
            continue
        if not os.path.isfile(vol1):
            # replace = input('There is already a file with the name {}.'. format())
            print('File does not exist')
            continue
        break
    while True:
        vol2 = input('Enter full path of query volume MRC file: ')
        if vol2[-4:] != '.mrc':
            print('Please enter an MRC file')
            continue
        if not os.path.isfile(vol2):
            # replace = input('There is already a file with the name {}.'. format())
            print('File does not exist')
            continue
        break

    while True:
        output_vol = input('Enter full path of output aligned volume MRC file: ')
        if output_vol[-4:] != '.mrc':
            print('Please enter an MRC file')
            continue
        if os.path.isfile(output_vol):
            break_flag = False
            while True:
                replace = input(
                    'There is already a file with the name {}. Do you want to replace it [y/n]?: '.format(output_vol))
                if replace.lower().startswith('y'):
                    break_flag = True
                    break
                if replace.lower().startswith('n'):
                    break
            if break_flag:
                break
        break

    while True:
        downsample = input('Enter the downsampled size in pixels (default {}): '.format(DEFAULT_DOWNSAMPLE))
        if downsample == '':
            downsample = DEFAULT_DOWNSAMPLE
            break
        try:
            downsample = int(downsample)
            if downsample < 1:
                print("Downsample size must be a positive integer.")
            else:
                break
        except ValueError:
            print("Downsample size must be a positive integer.")

    while True:
        n_projs = input('Enter the number of projections (default {}): '.format(DEFAULT_N_PROJS))
        if n_projs == '':
            n_projs = DEFAULT_N_PROJS
            break
        try:
            n_projs = int(n_projs)
            if n_projs < 1:
                print("Number of projections must be a positive integer.")
            else:
                break
        except ValueError:
            print("Number of projections must be a positive integer.")
            
    # Ask to skip refinement of alignment parameters
    while True:
        no_refine = input('Do you want to refine alignment parameters [y/n]')

        if any(no_refine.lower() == f for f in ["yes", 'y', '1', 'ye']):
            no_refine = False
            break
        elif any(no_refine.lower() == f for f in ['no', 'n', '0']):
            no_refine = True
            break
        else:
            print("Please choose y/n.")
            
    # Check if user want to save parameters
    while True:
        save_params = input('Do you want to save output parameters [y/n]: ')
        if save_params.lower().startswith('y'):
            # If he does, ask for file
            while True:
                output_parameters = input('Enter full path of output parameters txt file: ')
                if output_parameters[-4:] != '.txt':
                    print('Please enter an txt file')
                    continue
                if os.path.isfile(output_parameters):
                    break_flag = False
                    while True:
                        replace = str(input(
                            'There is already a file with the name {}. Do you want to replace it [y/n]?: '.format(
                                output_parameters)))
                        if replace.lower().startswith('y'):
                            break_flag = True
                            break
                        if replace.lower().startswith('n'):
                            break
                    if break_flag:
                        break
                break
            break
        if save_params.lower().startswith('n'):
            output_parameters = None
            break

    verbose = 0
    while verbose == 0:
        verbose_in = input('Display detailed progress? [y/n]: ')
        if verbose_in.strip().lower().startswith('y'):
            verbose = True
        elif verbose_in.strip().lower().startswith('n'):
            verbose = False
            break
        else:
            print("Please choose y/n.")

    return vol1, vol2, output_vol, downsample, n_projs, no_refine, output_parameters, verbose


def check_for_newer_version():
    """
    This function checks whether there is a newer version of kltpicker
    available on PyPI. If there is, it issues a warning.

    """
    name = 'EMalign'
    # Use pip to try and install a version of kltpicker which does not exist.
    # In answer, you get all available versions. Find the newest one.
    latest_version = str(
        subprocess.run([sys.executable, '-m', 'pip', 'install', '%s==random' % name], capture_output=True, text=True))
    latest_version = latest_version[latest_version.find('(from versions:') + 15:]
    latest_version = latest_version[:latest_version.find(')')]
    latest_version = latest_version.replace(' ', '').split(',')[-1]

    if latest_version == 'none':  # Got an unexpected response.
        pass
    else:  # Use pip to determine the installed version.
        import pkg_resources  # part of setuptools
        current_version = pkg_resources.require(name)[0].version
        if latest_version != current_version:
            print(
                "NOTE: you are running an old version of %s (%s). A newer version (%s) is available, please upgrade." % (
                    name, current_version, latest_version))
