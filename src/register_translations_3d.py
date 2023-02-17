#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 21:00:50 2022

@author: yoelsh
"""

import numpy as np
from numpy import fft
from src.reshift_vol import reshift_vol
import src.reshift_vol
from scipy.optimize import minimize
import pyfftw


class fftw_data_class:
    def __init__(self, in_data, num_threads=1):
        n = in_data.shape[0]
        n2 = n//2 + 1

        if in_data.dtype == np.float32:
            real_type = np.float32
            complex_type = np.complex64
        else:
            real_type = np.float64
            complex_type = np.complex128

        self.in_array_f = pyfftw.empty_aligned((n,n,n),dtype=real_type)
        self.in_array_b = pyfftw.empty_aligned((n,n,n2),dtype=complex_type)
        self.fftw_object_f = pyfftw.builders.rfftn(self.in_array_f)
        self.fftw_object_b = pyfftw.builders.irfftn(self.in_array_b)


# %%
def eval3Dshift(X, vol1, vol2, reshift_cache):
    dx = X[0]
    dy = X[1]
    dz = X[2]
    vol2_s = reshift_vol(vol2, np.array([dx, dy, dz]), reshift_cache)
    c = np.mean(np.corrcoef(vol1.ravel(), vol2_s.ravel(), rowvar=False)[0, 1:]).astype('float64')
    e = 1.0 - c
    return e


# %%
def refine3DshiftBFGS(vol1, vol2, estdx):
    # Create initial guess vector
    X0 = np.array([estdx[0].real, estdx[1].real, estdx[2].real]).astype('float64')
    # BFGS optimization:
        
    reshift_cache = src.reshift_vol.fftw_data_class(vol1)
    res = minimize(eval3Dshift, X0, args=(vol1, vol2, reshift_cache), 
                   method='BFGS', tol=1e-3, 
                   options={'gtol': 1e-1, 'disp': False})
    X = res.x
    estdx = np.array([X[0], X[1], X[2]])
    return estdx


# %%
def register_translations_3d(vol1, vol2, fftw_object=None):
    # REGISTER_TRANSLATIONS_3D  Estimate relative shift between two volumes.
    # register_translations_3d(vol1,vol2,refdx)
    #   Estimate the relative shift between two volumes vol1 and vol2 to 
    #   integral pixel accuracy. The function uses phase correlation to 
    #   estimate the relative shift to within one pixel accuray.
    #
    #   Input parameters:
    #   vol1,vol2 Two volumes to register. Volumes must be odd-sized.
    #   refidx    Two-dimensional vector with the true shift between the images,
    #             used for debugging purposes. (Optional)
    #   Output parameters
    #   estidx  A two-dimensional vector of how much to shift vol2 to aligh it
    #           with vol1. Returns -1 on error.
    #   err     Difference between estidx and refidx.

    # Take Fourer transform of both volumes and compute the phase correlation
    # factors.
    
    if fftw_object is None:
        fftw_object=fftw_data_class(vol1)
    
    #hats1 = fft.fftn(vol1)  # Compute centered Fourier transform
    #hats2 = fft.fftn(vol2)

    fftw_object.in_array_f[:] = vol1[:]
    hats1 = (fftw_object.fftw_object_f(fftw_object.in_array_f)).copy()
    fftw_object.in_array_f[:] = vol2[:]
    hats2 = (fftw_object.fftw_object_f(fftw_object.in_array_f)).copy()

    tmp1 = hats1 * np.conj(hats2)
    tmp2 = abs(tmp1)
    bool_idx = tmp2 < 2*np.finfo(vol1.dtype).eps
    tmp2[bool_idx] = 1 # Avoid division by zero. 
                       # The numerator for these indices is small anyway.
    rhat =  tmp1 / tmp2
    

    # Compute the relative shift between the images to to within 1 pixel
    # accuracy.
    #r = fft.ifftn(rhat).real
    fftw_object.in_array_b[:] = rhat[:]
    r = fftw_object.fftw_object_b(fftw_object.in_array_b)
    ii = np.argmax(r)

    # Find the center
    n = np.size(vol1, 0)
    cX = np.fix(n / 2)
    cY = np.fix(n / 2)
    cZ = np.fix(n / 2)
    [sX, sY, sZ] = np.unravel_index(ii, np.shape(r))
    if sX>cX:
        sX = sX - n
    if sY>cY:
        sY = sY - n
    if sZ>cZ:
        sZ = sZ - n

    estdx = [-sX,-sY,-sZ]

    # No need to refine tranlations
    return np.array(estdx)

    # bestdx = refine3DshiftBFGS(vol1, vol2, estdx)

    # return bestdx
