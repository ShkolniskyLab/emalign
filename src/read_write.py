# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import mrcfile
import numpy as np


def write_mrc(file_path, x):
    # For now it is transposed, when moving to C aligned this should be removed
    with mrcfile.new(file_path, overwrite=True) as mrc_fh:
        mrc_fh.set_data(x.astype('float32').T)
    return


def read_mrc(file_path):
    return np.ascontiguousarray(mrcfile.open(file_path).data.T)

