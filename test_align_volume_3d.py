#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:55:48 2022

@author: yaelharpaz1
"""

import time
import numpy as np
import logging
from src.read_write import read_mrc
from src.common_finufft import cryo_downsample
from src.SymmetryGroups import genSymGroup
from src.rand_rots import rand_rots
from src.align_volumes_3d import align_volumes
from src.fastrotate3d import fastrotate3d
from src.reshift_vol import reshift_vol_int


# Test for volume alignment
#np.random.seed(1337)
np.random.seed(2021)

# Read molecule:
#vol = read_mrc('0825_C6.mrc')
#sym = 'C6'

vol = read_mrc('10280_C1.mrc')
sym = 'C1'

#vol = read_mrc('9203_D3.mrc')
#sym = 'D3'

#vol = read_mrc('4179_T.mrc')
#sym = 'T'

#vol = read_mrc('24494_I.mrc')
#sym = 'I'

#vol = read_mrc('22658_O.mrc')
#sym = 'O'


#out_shape = (128,128,128)
#vol = cryo_downsample(vol,out_shape)

# s=[1.1,1.2,1.3]
# import src.reshift_vol
# t1 = time.perf_counter()
# svol1=src.reshift_vol.reshift_vol(vol, s)
# t2 = time.perf_counter()
# print("time ref = ",str(t2-t1))

# t1 = time.perf_counter()
# svol2=src.reshift_vol.reshift_vol_rfft(vol, s)
# t2 = time.perf_counter()
# print("time fftn = ",str(t2-t1))

# print(np.linalg.norm(svol1[:] - svol2[:])/np.linalg.norm(svol1[:]))

np.random.seed(1338)
R_true = rand_rots(1).reshape((3,3))
#R_true = mat_to_npy('trueR_test3d_forVol')

vol_c = np.copy(vol)
vol_rotated = fastrotate3d(vol_c, R_true)
vol_rotated =  np.flip(vol_rotated, axis=2)
vol_rotated = reshift_vol_int(vol_rotated, np.array([-5, 0 ,0]))

# Alignment algorithm:
G = genSymGroup(sym)
class Struct:
    '''
    Used to pass optimal paramters to the alignment function
    '''
    pass

opt = Struct()
opt.sym = sym
opt.Nprojs = 30
opt.G = G
opt.downsample = 64
opt.trueR = R_true
opt.no_refine = False
#opt.only_estimate_rotation = True

logging.basicConfig(level=logging.INFO, format='%(message)s')
t_start = time.perf_counter()
bestR, bestdx, reflect, vol2aligned, bestcorr = align_volumes(vol,vol_rotated,verbose=1,opt=opt)
t = time.perf_counter() - t_start
print("Timing = "+str(t))
