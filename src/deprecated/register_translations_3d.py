#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:33:25 2021

@author: yaelharpaz1
"""

import numpy as np
from numpy import fft
from src.reshift_vol import reshift_vol
from scipy.optimize import minimize


# %%
def eval3Dshift(X, vol1, vol2):
    dx = X[0]
    dy = X[1]
    dz = X[2]
    vol2_s = reshift_vol(vol2, np.array([dx, dy, dz]))
    c = np.mean(np.corrcoef(vol1.ravel(), vol2_s.ravel(), rowvar=False)[0, 1:]).astype('float64')
    e = 1.0 - c
    return e


# %%
def refine3DshiftBFGS(vol1, vol2, estdx):
    # Create initial guess vector
    X0 = np.array([estdx[0].real, estdx[1].real, estdx[2].real]).astype('float64')
    # BFGS optimization:
    res = minimize(eval3Dshift, X0, args=(vol1, vol2), method='BFGS', tol=1e-2,
                   options={'gtol': 1e-2, 'disp': False})
    X = res.x
    estdx = np.array([X[0], X[1], X[2]])
    return estdx


# %%
def register_translations_3d(vol1, vol2):
    # REGISTER_TRANSLATIONS_3D  Estimate relative shift between two volumes.
    # register_translations_3d(vol1,vol2,refdx)
    #   Estimate the relative shift between two volumes vol1 and vol2 to subpixel
    #   accuracy. The function uses phase correlation to estimate the relative
    #   shift to within one pixel accuray, and then refines the estimation
    #   using Newton iterations.
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
    hats1 = fft.fftshift(fft.fftn(fft.ifftshift(vol1)))  # Compute centered Fourier transform
    hats2 = fft.fftshift(fft.fftn(fft.ifftshift(vol2)))
    rhat = hats1 * np.conj(hats2) / (abs(hats1 * np.conj(hats2)))
    rhat[np.isnan(rhat)] = 0
    rhat[np.isinf(rhat)] = 0
    n = np.size(vol1, 0)
    # ll = np.fix(n/2)
    # freqrng = np.arange(-ll,n-ll)
    # Compute the relative shift between the images to to within 1 pixel
    # accuracy.
    # mm is a window function that can be applied to the volumes before
    # computing their relative shift. Experimenting with the code shows that
    # windowing does not improve accuracy.
    mm = 1  # Windowing does not improve accuracy.
    r = fft.fftshift(fft.ifftn(fft.ifftshift(rhat * mm))).real
    ii = np.argmax(r)
    # Find the center
    cX = np.fix(n / 2)
    cY = np.fix(n / 2)
    cZ = np.fix(n / 2)
    [sX, sY, sZ] = np.unravel_index(ii, np.shape(r))
    estdx = [cX - sX, cY - sY, cZ - sZ]

    # No need to refine tranlations
    return np.array(estdx)

    # bestdx = refine3DshiftBFGS(vol1, vol2, estdx)

    # return bestdx
