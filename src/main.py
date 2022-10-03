import shutil
import warnings
from sys import exit, argv
from src.emalign_input import check_for_newer_version, get_args, parse_args
from src.AlignVolumes3d import AlignVolumes
from src.read_write import read_mrc, copy_and_rename
from src.gentestdata import gentestdata
import mrcfile
#from src.common_finufft import cryo_downsample
#import os
#import shutil

warnings.filterwarnings("ignore")

# Check if CuPy is installed and we have GPU devices


def main():
    # Get user arguments:
    user_input = argv
    if len(user_input) > 1:  # User entered arguments. Use command line mode.
        args = parse_args()
        # Check if user entered the mandatory arguments: input and output
        # directory and particle size. If not, exit.
        if args.version:
            import pkg_resources  # part of setuptools
            version = pkg_resources.require('EMalign')[0].version
            print('EMalign {}'.format(version))
            exit()
            
        if args.make_test_data:
            emdID = 2660
            ref_mrc_filename = "map_ref_{0}.mrc".format(emdID)
            transformed_mrc_filename = "map_transformed_{0}.mrc".format(emdID)
            print("Generating test data...")
            gentestdata(ref_mrc_filename, transformed_mrc_filename, emdID, args.verbose)
            print("Test volume saved to " + ref_mrc_filename)
            print("Transform volume saved to "+ transformed_mrc_filename)
            exit()
            
        if args.vol1 is None or args.vol2 is None or args.output_vol is None:
            print(
                "Error: one or more of the following arguments are missing: vol1, vol2, output-vol. For help run kltpicker -h")
            exit()

    else:  # User didn't enter arguments, use interactive mode to get arguments.
        args = parse_args()  # Initiate args with default values.
        args.vol1, args.vol2, args.output_vol, args.downsample, args.n_projs, args.no_refine, args.output_parameters, args.verbose = get_args()

    # Check newer version
    try:
        check_for_newer_version()
    except:
        pass

    # Load volumes
    vol1 = read_mrc(args.vol1)
    vol2 = read_mrc(args.vol2)

    # If we decide to downsample them to the same size
    # n1 = vol1.shape[0]
    # n2 = vol2.shape[0]
    # if n1 < n2:
    #     vol2 = cryo_downsample(vol2, (n1, n1, n1))
    # elif n2 < n1:
    #     vol1 = cryo_downsample(vol1, (n2, n2, n2))

    # create params
    class Struct:
        pass

    opt = Struct()
    opt.Nprojs = args.n_projs
    opt.downsample = args.downsample
    opt.no_refine = args.no_refine
    
    # Run
    bestR, bestdx, reflect, vol2aligned, bestcorr = AlignVolumes(vol1, vol2, args.verbose, opt)

    # Save
    # Copy vol2 to save header
    shutil.copyfile(args.vol2, args.output_vol)

    # Change and save
    mrc_fh = mrcfile.open(args.output_vol, mode='r+')
    mrc_fh.set_data(vol2aligned.astype('float32').T)
    mrc_fh.set_volume()
    mrc_fh.update_header_stats()
    mrc_fh.close()

    # Save parameters
    if args.output_parameters is not None:
        lines = ['reflect:\t{}'.format(reflect), 'correlation:\t{}'.format(bestcorr),
                 'estimated translation:\t{}'.format(bestdx), 'rotation:\n{}'.format(bestR)]
        with open(args.output_parameters, 'w') as f:
            f.writelines('\n'.join(lines))
