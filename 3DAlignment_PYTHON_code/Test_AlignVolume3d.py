#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:55:48 2022

@author: yaelharpaz1
"""

import numpy as np
from read_write import read_mrc
from dev_utils import npy_to_mat, mat_to_npy
from common_finufft import cryo_downsample
from SymmetryGroups import genSymGroup
from rand_rots import rand_rots
from AlignVolumes3d import AlignVolumes
from fastrotate3d import fastrotate3d
from reshift_vol import reshift_vol


# Test for volume alignment

# Read molecule:
vol = read_mrc('0825_C6.mrc')
sym = 'C6'

#vol = read_mrc('10280_C1.mrc')
#sym = 'C1'

#vol = read_mrc('9203_D3.mrc')
#sym = 'D3'

#vol = read_mrc('4179_T.mrc')
#sym = 'T'

#vol = read_mrc('24494_I.mrc')
#sym = 'I'

#vol = read_mrc('22658_O.mrc')
#sym = 'O'


out_shape = (129,129,129)
vol = cryo_downsample(vol,out_shape)

np.random.seed(1337)
R_true = rand_rots(1).reshape((3,3))
#R_true = mat_to_npy('trueR_test3d_forVol')

vol_c = np.copy(vol)
vol_rotated = fastrotate3d(vol_c, R_true)
vol_rotated =  np.flip(vol_rotated, axis=2)  
vol_rotated = reshift_vol(vol_rotated, np.array([-5, 0 ,0]))

# Alignment algorithm:
G = genSymGroup(sym)    
class Struct:
    pass
opt = Struct() 
opt.sym = sym   
opt.Nref = 30
opt.G = G
opt.downsample = 48
opt.trueR = R_true

bestR, bestdx, reflect, vol2aligned, bestcorr = AlignVolumes(vol,vol_rotated,1,opt)
