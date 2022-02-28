from pathlib import Path
import sys
import subprocess
import argparse
import os

DEFAULT_DOWNSAMPLE = 64
DEFAULT_N_PROJS = 30


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v1', '--vol1', help='Full path of volume1 mrc input file.')
    parser.add_argument('-v2', '--vol2', help='Full path of volume1 mrc input file.')
    parser.add_argument('-o', '--output-vol', help='Full path of mrc output file.', type=check_file_exists)
    parser.add_argument('--downsample', help='Use to set downsample size', default=DEFAULT_DOWNSAMPLE, type=check_positive_int)
    parser.add_argument('--n-projs', help='Use to set number of projections', default=DEFAULT_N_PROJS, type=check_positive_int)
    parser.add_argument('--output-parameters', help='Full path of txt output file for parameters.', default=None, type=check_file_exists)
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose. Choose this to display outputs during runtime.', default=False)

    # Version
    parser.add_argument('--version', help='Check version', action='store_true', default=False)

    args = parser.parse_args()
    return args


def check_positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue



def check_file_exists(output_file):
    if os.path.isfile(output_file):
        raise argparse.ArgumentTypeError("There is already a file with the name %s." % output_file)
    output_dir, _ = os.path.split(output_file)
    if not os.path.isdir(output_dir):
        raise argparse.ArgumentTypeError(
            'Directory %s does not exist. Please specify an existing directory.' % output_dir)
    return output_file


def get_args():
    print("\n")
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
                replace = input('There is already a file with the name {}. Do you want to replace it [y/n]?: '. format(output_vol))
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
        if n_projs== '':
            n_projs= DEFAULT_N_PROJS
            break
        try:
            n_projs= int(n_projs)
            if n_projs < 1:
                print("Number of projections must be a positive integer.")
            else:
                break
        except ValueError:
            print("Number of projections must be a positive integer.")

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

    return vol1, vol2, output_vol, downsample, n_projs, output_parameters, verbose


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
