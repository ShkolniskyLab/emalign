import warnings
from sys import exit, argv
import numpy as np
from src.emalign_input import check_for_newer_version, get_args, parse_args
from src.AlignVolumes3d import AlignVolumes
from src.read_write import read_mrc, write_mrc
from src.common_finufft import cryo_downsample

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
        if args.vol1 is None or args.vol2 is None or args.output_vol is None:
            print(
                "Error: one or more of the following arguments are missing: vol1, vol2, output-vol. For help run kltpicker -h")
            exit()

    else:  # User didn't enter arguments, use interactive mode to get arguments.
        args = parse_args()  # Initiate args with default values.
        args.vol1, args.vol2, args.output_vol, args.downsample, args.n_projs, args.verbose = get_args()

    # Check newer version
    try:
        check_for_newer_version()
    except:
        pass

    # Load volumes
    vol1 = read_mrc(args.vol1)
    vol2 = read_mrc(args.vol2)

    # Testing purpose
    n1 = vol1.shape[0]
    n2 = vol2.shape[0]
    if n1 < n2:
        vol2 = cryo_downsample(vol2,(n1, n1, n1))
    elif n2 < n1:
        vol1 = cryo_downsample(vol1, (n2, n2, n2))

    # create params
    class Struct:
        pass

    opt = Struct()
    opt.Nref = args.n_projs
    opt.downsample = args.downsample

    # Run
    bestR, bestdx, reflect, vol2aligned, bestcorr = AlignVolumes(vol1, vol2, args.verbose, opt)

    # Save
    write_mrc(args.output_vol, vol2aligned)
