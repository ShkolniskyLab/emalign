#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 11:54:51 2021

@author: yaelharpaz1
"""

import numpy as np
from numpy import fft
from numpy import linalg as LA

#%% 
def reshift_vol(vol,s):
    # Shift the volume given by im by the vector s using trigonometric
    # interpolation. The volume im is of nxnxn, where n can be odi or even. The vector
    # s\in \mathbb{R}^{3} contains the hshifts.
    #
    # Example: Shift the volume vol by 1 pixel in the x direction, 2 in the y
    # direction, and 3 in the z direction
    #
    #       s = [1 2 3];
    #       vols=shift_vol(vol,s);
    #
    # NOTE: I don't know if s=[0 0 1 ] shifts up or down, but this can be easily checked. Same issue for the other directions.  
    if vol.ndim != 3:
        raise ValueError("Input must be a 3D volume")
    if (np.size(vol,0) != np.size(vol,1)) or (np.size(vol,1) != np.size(vol,2)):
        raise ValueError("All three dimension of the input must be equal")   
    n = np.size(vol,0)
    ll = np.fix(n/2)
    freqrng = np.arange(-ll,n-ll)
    [omega_x,omega_y,omega_z] = np.meshgrid(freqrng,freqrng,freqrng,indexing='ij')  
    omega_x = 2*np.pi*omega_x/n 
    omega_y = 2*np.pi*omega_y/n
    omega_z = 2*np.pi*omega_z/n   
    phase_x = np.exp(1j*omega_x*s[0])
    phase_y = np.exp(1j*omega_y*s[1])
    phase_z = np.exp(1j*omega_z*s[2])   
    # Force conjugate symmetry. Otherwise this frequency component has no
    # corresponding negative frequency to cancel out its imaginary part.
    if np.mod(n,2) == 0:
        phase_x[0,:,:] = np.real(phase_x[0,:,:])
        phase_y[:,0,:] = np.real(phase_y[:,0,:])
        phase_z[:,:,0] = np.real(phase_z[:,:,0])        
    phases = phase_x*phase_y*phase_z
    pim = fft.fftshift(fft.fftn(fft.ifftshift(vol)))
    pim = pim*phases
    svol = fft.fftshift(fft.ifftn(fft.ifftshift(pim)).real) 
    if LA.norm(np.imag(svol[:]))/LA.norm(svol[:]) > 1.0e-8:
        raise ValueError("Large imaginary components")
    svol = np.real(svol)
    return svol