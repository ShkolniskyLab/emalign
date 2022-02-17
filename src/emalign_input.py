from pathlib import Path
import sys
import subprocess
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v1', '--vol1', help='Full path of volume1 mrc input file.')
    parser.add_argument('-v2', '--vol2', help='Full path of volume1 mrc input file.')
    parser.add_argument('-o', '--output-vol', help='Full path of mrc output file.', type=check_file_exists)
    parser.add_argument('--downsample', help='Use to set downsample size', default=64, type=check_positive_int)
    parser.add_argument('--n-projs', help='Use to set number of projections', default=30, type=check_positive_int)
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose. Choose this to display outputs during runtime.', default=False)
    # Add more output options
    # parser.add_argument('--max-processes',
    #                     help='Limit the number of concurrent processes to run. -1 to let the program choose.',
    #                     type=check_positive_int_or_all, default=-1)
    # parser.add_argument('--version', help='Check version', action='store_true', default=False)

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
        input_dir = Path(input('Enter full path of micrographs MRC files: '))
        num_files = len(list(input_dir.glob("*.mrc")))
        if num_files > 0:
            print("Found %i MRC files." % len(list(input_dir.glob("*.mrc"))))
            break
        elif not input_dir.is_dir():
            print("%s is not a directory." % input_dir)
        else:
            print("Could not find any files in %s." % input_dir)

    while True:
        particle_size = input('Enter the particle size in pixels: ')
        try:
            particle_size = int(particle_size)
            if particle_size < 1:
                print("Particle size must be a positive integer.")
            else:
                break
        except ValueError:
            print("Particle size must be a positive integer.")

    only_do_unfinished = False
    while True:
        output_path = input('Enter full path of output directory: ')
        output_dir = Path(output_path)
        if output_dir.is_file():
            print("There is already a file with the name you specified. Please specify a directory.")
        elif not output_path:
            print("Please specify a directory.")
        elif output_dir.parent.exists() and not output_dir.exists():
            while True:
                create_dir = input('Output directory does not exist. Create? (Y/N): ')
                if create_dir.strip().lower().startswith('y'):
                    Path.mkdir(output_dir)
                    break
                elif create_dir.strip().lower().startswith('n'):
                    print("OK, aborting...")
                    sys.exit(0)
                else:
                    print("Please choose Y/N.")
            break
        elif not output_dir.parent.exists():
            print('Parent directory %s does not exist. Please specify an existing directory.' % output_dir.parent)
        elif output_dir.is_dir():
            num_finished = check_output_dir(input_dir, output_dir, particle_size)
            if num_finished == 1:
                break
            elif num_finished == 2:
                print(
                    "The directory you specified contains coordinate files for all of the micrographs in the input directory. Aborting...")
                sys.exit()
            else:
                while True:
                    only_do_unfinished = input(
                        "The directory you specified contains coordinate files for some of the micrographs in the input directory. Run only on micrographs which have no coordinate file? (Y/N): ")
                    if only_do_unfinished.strip().lower().startswith('y'):
                        only_do_unfinished = True
                        break
                    elif only_do_unfinished.strip().lower().startswith('n'):
                        print("OK, aborting...")
                        sys.exit(0)
                    else:
                        print("Please choose Y/N.")
                break

    num_particles_to_pick = 0
    while num_particles_to_pick == 0:
        pick_all = input('Pick all particles? (Y/N): ')
        if pick_all.strip().lower().startswith('y'):
            num_particles_to_pick = -1
        elif pick_all.strip().lower().startswith('n'):
            while True:
                num_particles_to_pick = input('How many particles to pick: ')
                try:
                    num_particles_to_pick = int(num_particles_to_pick)
                    if num_particles_to_pick < 1:
                        print("Number of particles to pick must be a positive integer.")
                    else:
                        break
                except ValueError:
                    print("Number of particles to pick must be a positive integer.")
        else:
            print("Please choose Y/N.")

    num_noise_to_pick = -1
    while num_noise_to_pick == -1:
        pick_noise = input('Pick noise images? (Y/N): ')
        if pick_noise.strip().lower().startswith('n'):
            num_noise_to_pick = 0
        elif pick_noise.strip().lower().startswith('y'):
            while True:
                num_noise_to_pick = input('How many noise images to pick: ')
                try:
                    num_noise_to_pick = int(num_noise_to_pick)
                    if num_noise_to_pick < 1:
                        print("Number of noise images to pick must be a positive integer.")
                    else:
                        break
                except ValueError:
                    print("Number of particles to pick must be a positive integer.")
        else:
            print("Please choose Y/N.")

    use_asocem = -1
    while use_asocem == -1:
        pick_asocem = input('Do you want to use ASOCEM for contamination removal? (Y/N):')
        if pick_asocem.strip().lower().startswith('n'):
            use_asocem = 0
        elif pick_asocem.strip().lower().startswith('y'):
            use_asocem = 1
        else:
            print("Please choose Y/N.")

    if use_asocem:
        save_asocem_masks = -1
        while save_asocem_masks == -1:
            save_asocem_input = input('Do you want to save ASOCEM masks? (Y/N):')
            if save_asocem_input.strip().lower().startswith('n'):
                save_asocem_masks = 0
            elif save_asocem_input.strip().lower().startswith('y'):
                save_asocem_masks = 1
            else:
                print("Please choose Y/N.")

        change_params = -1
        while change_params == -1:
            change_params_input = input('Do you want to change ASOCEM default parameters? (Y/N):')
            if change_params_input.strip().lower().startswith('n'):
                change_params = 0
            elif change_params_input.strip().lower().startswith('y'):
                change_params = 1
            else:
                print("Please choose Y/N.")

        if change_params:
            # ASOCEM DS
            asocem_downsample = -1
            while asocem_downsample == -1:
                asocem_downsample_input = input('Enter ASOCEM downsample image size (should be a positive number):')
                try:
                    asocem_downsample = int(asocem_downsample_input)
                    if asocem_downsample < 1:
                        print("Downsample image size must be a positive integer.")
                    else:
                        break
                except ValueError:
                    print("Downsample image size must be a positive integer.")

            # ASOCEM area
            asocem_area = -1
            while asocem_area == -1:
                asocem_area_input = input('Enter ASOCEM covariance area size (should be a positive odd number):')
                try:
                    asocem_area = int(asocem_area_input)
                    if asocem_area < 1:
                        print("Covariance area size should be positive.")
                    elif asocem_area % 2 == 0:
                        print("Covariance area size should be odd.")
                    else:
                        break
                except ValueError:
                    print("Covariance area size must be a positive integer.")

        else:
            asocem_downsample = 600
            asocem_area = 5
    else:
        save_asocem_masks = 0
        asocem_downsample = 600
        asocem_area = 5

    verbose = 0
    while verbose == 0:
        verbose_in = input('Display detailed progress? (Y/N): ')
        if verbose_in.strip().lower().startswith('y'):
            verbose = True
        elif verbose_in.strip().lower().startswith('n'):
            verbose = False
            break
        else:
            print("Please choose Y/N.")

    while True:
        max_processes_in = input('Enter maximum number of concurrent processes (-1 to let the program decide): ')
        try:
            max_processes = int(max_processes_in)
            if max_processes < 1 and max_processes != -1:
                print(
                    "Maximum number of concurrent processes must be a positive integer (except -1 to let the program decide).")
            else:
                break
        except ValueError:
            print(
                "Maximum number of concurrent processes must be a positive integer (except -1 to let the program decide).")

    return input_dir, output_dir, particle_size, num_particles_to_pick, num_noise_to_pick, use_asocem, save_asocem_masks, asocem_downsample, asocem_area, verbose, max_processes, only_do_unfinished


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
