#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 11:54:51 2021

@author: yaelharpaz1
"""

import numpy as np
from numpy import fft
from numpy import linalg as LA
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


#%% 
# def reshift_vol(vol,s):
#     # Shift the volume given by im by the vector s using trigonometric
#     # interpolation. The volume im is of nxnxn, where n can be odi or even. The vector
#     # s\in \mathbb{R}^{3} contains the hshifts.
#     #
#     # Example: Shift the volume vol by 1 pixel in the x direction, 2 in the y
#     # direction, and 3 in the z direction
#     #
#     #       s = [1 2 3];
#     #       vols=shift_vol(vol,s);
#     #
#     # NOTE: I don't know if s=[0 0 1 ] shifts up or down, but this can be easily checked. Same issue for the other directions.  
#     if vol.ndim != 3:
#         raise ValueError("Input must be a 3D volume")
#     if (np.size(vol,0) != np.size(vol,1)) or (np.size(vol,1) != np.size(vol,2)):
#         raise ValueError("All three dimension of the input must be equal")   
#     n = np.size(vol,0)
#     ll = np.fix(n/2)
#     freqrng = np.arange(-ll,n-ll)
#     [omega_x,omega_y,omega_z] = np.meshgrid(freqrng,freqrng,freqrng,indexing='ij')  
#     omega_x = 2*np.pi*omega_x/n 
#     omega_y = 2*np.pi*omega_y/n
#     omega_z = 2*np.pi*omega_z/n   
#     phase_x = np.exp(1j*omega_x*s[0])
#     phase_y = np.exp(1j*omega_y*s[1])
#     phase_z = np.exp(1j*omega_z*s[2])   
#     # Force conjugate symmetry. Otherwise this frequency component has no
#     # corresponding negative frequency to cancel out its imaginary part.
#     if np.mod(n,2) == 0:
#         phase_x[0,:,:] = np.real(phase_x[0,:,:])
#         phase_y[:,0,:] = np.real(phase_y[:,0,:])
#         phase_z[:,:,0] = np.real(phase_z[:,:,0])        
#     phases = phase_x*phase_y*phase_z
#     pim = fft.fftshift(fft.fftn(fft.ifftshift(vol)))
#     pim = pim*phases
#     svol = fft.fftshift(fft.ifftn(fft.ifftshift(pim)).real) 
#     if LA.norm(np.imag(svol[:]))/LA.norm(svol[:]) > 1.0e-8:
#         raise ValueError("Large imaginary components")
#     svol = np.real(svol)
#     return svol

#%% 
def reshift_vol(vol,s,fftw_data=None):
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
    freqrngxy = np.arange(0,n)
    freqrngz = np.arange(0,n//2+1)
    [omega_x,omega_y,omega_z] = np.meshgrid(freqrngxy,freqrngxy,freqrngz,indexing='ij')  
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
      
    
    if fftw_data is None:
        fftw_data = fftw_data_class(vol)
    
    fftw_data.in_array_f[:] = vol[:]
    pim = fftw_data.fftw_object_f(fftw_data.in_array_f)
    pim = pim*phases
    svol = fftw_data.fftw_object_b(pim)
  
    if LA.norm(np.imag(svol[:]))/LA.norm(svol[:]) > 1.0e-8:
        raise ValueError("Large imaginary components")
  
    return svol

# def test():
#     import time
#     #vol = np.random.rand(256,256,256)
#     vol = np.random.rand(256,256,256).astype(np.float32)
#     start_time = time.perf_counter()
#     vol1=reshift_vol(vol,[-10,5,6])
#     total_time = time.perf_counter() - start_time
#     print("ref timeing ",total_time)
#     start_time = time.perf_counter()
#     vol2=reshift_vol_opt(vol,[-10,5,6])
#     total_time = time.perf_counter() - start_time
#     print("optimized timing ",total_time)
#     print(np.allclose(vol1,vol2))
#     print(np.linalg.norm(vol1.ravel()-vol2.ravel())/np.linalg.norm(vol1.ravel()))    
    
#test()
