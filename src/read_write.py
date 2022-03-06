# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import mrcfile
import numpy as np
import shutil
import os


def write_mrc(file_path, x):
    # For now it is transposed, when moving to C aligned this should be removed
    with mrcfile.new(file_path, overwrite=True) as mrc_fh:
        mrc_fh.set_data(x.astype('float32').T)
    return


def read_mrc(file_path):
    return np.ascontiguousarray(mrcfile.open(file_path).data.T)


def copy_and_rename(source_file, target_file):
    # Copy vol2 first
    target_destination, target_name = os.path.split(target_file)
    source_destination, source_name = os.path.split(source_file)
    intermidiate_file = os.path.join(target_destination, source_name)
    if os.path.isfile(intermidiate_file):
        # If the target directory has a file with this name create a temporary directory
        i = 0
        while True:
            tmp_dir = os.path.join(target_destination, 'tmp{}'.format(i))
            if os.path.isdir(tmp_dir):
                i += 1
            else:
                os.mkdir(tmp_dir)
                break

        # Copy file to new dir and rename it
        intermidiate_file = os.path.join(tmp_dir, source_name)
        shutil.copy(source_file, tmp_dir)
        os.rename(intermidiate_file, os.path.join(tmp_dir, target_name))

        # Copy from tmp dir to the real dir
        shutil.copy(os.path.join(tmp_dir, target_name), target_destination)

        # Delete intermidiate file and directory
        shutil.rmtree(tmp_dir)
    else:
        # Copy and rename
        shutil.copy(source_file, target_destination)
        os.rename(intermidiate_file, os.path.join(target_destination, target_name))
