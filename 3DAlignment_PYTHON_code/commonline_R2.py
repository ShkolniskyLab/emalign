#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 17:04:47 2021

@author: yaelharpaz1
"""
import numpy as np
import math 
from numpy import linalg as LA

def commonline_R2(Ri,Rj,L):
    Ri = np.transpose(Ri)
    Rj = np.transpose(Rj)
    
    Ri3 = Ri[:,2]
    Rj3 = Rj[:,2]
    
    clvec = np.array([[Ri3[1]*Rj3[2] - Ri3[2]*Rj3[1]],
                      [Ri3[2]*Rj3[0] - Ri3[0]*Rj3[2]],
                      [Ri3[0]*Rj3[1] - Ri3[1]*Rj3[0]]])
    
    # No need to normalize as the normalization does not affect the atan2 below.
    
    cij = (np.transpose(Ri)).dot(clvec)
    cji = (np.transpose(Rj)).dot(clvec)
    
    alphaij = math.atan2(cij[1], cij[0])
    alphaji = math.atan2(cji[1], cji[0])
    
    PI = 4*math.atan(1.0)
    alphaij = alphaij + PI
    alphaji = alphaji +PI
    
    l_ij = alphaij/(2*PI)*L
    l_ji = alphaji/(2*PI)*L
    
    l_ij = int(round(l_ij) % L)
    l_ji = int(round(l_ji) % L)
    return l_ij, l_ji

def cryo_normalize(pf):
    """
    Normalize a dataset of Fourier rays so that each ray has energy 1.
    pf is a 3D array of the Fourier transform of the projections. 
    pf(:,:,k) is the polar Fouier transform of the k'th projection.    
    """
    n_proj = 1
    if pf.ndim == 3:   
       n_proj = np.size(pf,2) 
    n_theta = np.size(pf,1)
    # create a copy of the data for normalization purposes
    pf2 = pf
    
    for k in range(n_proj):
        for j in range(n_theta):
            nr = LA.norm(pf[:,j,k])
            if nr < 1.0e-13:
                Warning.warn('Ray norm is close to zero. k=%d  j=%d' %(k, j))
            pf2[:,j,k] = pf2[:,j,k] / nr
            
    return pf2