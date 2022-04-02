#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 01:25:14 2022

@author: yaelharpaz1
"""


import numpy as np
from cryo_fetch_emdID import cryo_fetch_emdID
from common_finufft import cryo_downsample
from rand_rots import rand_rots
from fastrotate3d import fastrotate3d
from reshift_vol import reshift_vol
import logging


def gentestdata(emdID,verbose=1):
    """
    gentestdata  generates test volumes for the function AlignVolumes in 
    AlignVolumes3d.
    
    This function fetchs the map file (MRC format) with the given emdID 
    (integer) from EMDB.
    Generates a volume demonstrating the 3D structure from the retrived map 
    file. Generates an additional rotated and shifted volume, and saves the 
    two volumes.
    """
    np.random.seed(2021)
    
    logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger()
    if verbose == 0 : logger.disabled = True  
    
    # Read molecule:
    logger.info('Downloading the volume from EMDB')
    vol = cryo_fetch_emdID('0825',verbose) 
    logger.info('The volume was downloaded')
    
    n_ds = 129
    logger.info('Downsampling volume to %i pixels', n_ds)
    vol = cryo_downsample(vol,(n_ds, n_ds, n_ds)).astype('float64')

    # Rotate and shift the volume:
    logger.info('Generating a rotated and shifted volume')
    R_true = rand_rots(1).reshape((3,3))
    vol_rotated = fastrotate3d(vol.copy(), R_true)
    shift = np.array([-5, 3 ,0])
    vol_rotated = reshift_vol(vol_rotated.copy(), shift)
    
    logger.info('Ground truth rotation:')
    logger.info('%.4f %.4f %.4f', R_true[0,0], R_true[0,1], R_true[0,2])
    logger.info('%.4f %.4f %.4f', R_true[1,0], R_true[1,1], R_true[1,2])
    logger.info('%.4f %.4f %.4f', R_true[2,0], R_true[2,1], R_true[2,2])
    
    logger.info('Ground truth translation: [%.3f, %.3f, %.3f]',shift[0], shift[1], shift[2])
    
    np.save('vol_'+emdID, vol)
    np.save('vol_rotated_'+emdID, vol_rotated)
    
    return vol, vol_rotated
    