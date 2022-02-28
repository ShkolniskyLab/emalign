#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:33:25 2021

@author: yaelharpaz1
"""

import numpy as np
#import math
from numpy import fft
from pyfftw.interfaces.numpy_fft import ifftn, ifft2
from numpy import linalg as LA
from dev_utils import npy_to_mat, mat_to_npy


def E3(deltax,rho,N,idx):  
    # Take as an input the estimated shift deltax and the measured phase
    # difference between the signals, and check how well the phases induced by
    # deltax agree with the measured phases rho. That is, return the difference
    # between the induced phases and the measured ones.
    #
    # Only the phases whose index is given by idx are considered.
    # The sum of the squared absolute value of E3, that is,
    #   sum(abs(E3(x,rhat(idx),(n-1)./2,idx)).^2)
    # is the L2 error in the phases. This is the expression we minimize (over
    # deltax) to find the relative shift between the images.
    #
    # Yoel Shkolnisky, January 2014.
    ll = np.fix(N/2)
    freqrng = np.arange(-ll,N-ll) 
    [X,Y,Z] = np.meshgrid(freqrng,freqrng,freqrng,indexing='ij')
    y = (np.exp(2*np.pi*1j*(X.T.ravel()[idx].T*deltax[0]+Y.T.ravel()[idx].T*deltax[1]+Z.T.ravel()[idx].T*deltax[2])/N)-rho)  
    return y


#%%
def eval3Dshift(X,vol1,vol2):
    dx = X[0]
    dy = X[1]
    dz = X[2]
    vol2_s = reshift_vol(vol2.copy(),np.array([dx,dy,dz]))
    c = np.mean(np.corrcoef(vol1.ravel(),vol2_s.ravel(),rowvar=False)[0,1:])
    e = 1-c
    return e

#%%
def refine3DshiftBFGS(vol1,vol2,estdx):
    # Create initial guess vector
    X0 = np.array([estdx[0].real, estdx[1].real, estdx[2].real]).astype('float64')
    # BFGS optimization:
    res = minimize(eval3Dshift, X0, args=(vol1,vol2), method='BFGS', tol=1e-4)
    X = res.x
    estdx = np.array([X[0], X[1], X[2]])
    return estdx

#%%
def register_translations_3d(vol1,vol2):
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
    hats1 = fft.fftshift(fft.fftn(fft.ifftshift(vol1))) # Compute centered Fourier transform
    hats2 = fft.fftshift(fft.fftn(fft.ifftshift(vol2)))
    rhat = hats1 * np.conj(hats2) / (abs(hats1 * np.conj(hats2)))
    rhat[np.isnan(rhat)] = 0
    rhat[np.isinf(rhat)] = 0   
    n = np.size(vol1,0)
    #ll = np.fix(n/2)
    #freqrng = np.arange(-ll,n-ll)
    # Compute the relative shift between the images to to within 1 pixel
    # accuracy.
    # mm is a window function that can be applied to the volumes before
    # computing their relative shift. Experimenting with the code shows that
    # windowing does not improve accuracy.
    mm = 1 # Windowing does not improve accuracy.
    r = fft.fftshift(fft.ifftn(fft.ifftshift(rhat*mm))).real
    ii = np.argmax(r)    
    # Find the center
    cX = np.fix(n/2) 
    cY = np.fix(n/2) 
    cZ = np.fix(n/2)     
    [sX,sY,sZ] = np.unravel_index(ii,np.shape(r))
    estdx = [cX-sX,cY-sY,cZ-sZ]

    bestdx = refine3DshiftBFGS(vol1,vol2,estdx)
    
    return bestdx
        