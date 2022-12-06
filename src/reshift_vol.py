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
        #n2 = n//2+1

        #if in_data.dtype == np.float32:
        #    real_type = np.float32
        #    complex_type = np.complex64
        #else:
        #    real_type = np.float64
        #    complex_type = np.complex128

        #self.in_array_f = pyfftw.empty_aligned((n,n,n),dtype=real_type)
        #self.in_array_b = pyfftw.empty_aligned((n,n,n2),dtype=complex_type)
        #self.fftw_object_f = pyfftw.builders.rfftn(self.in_array_f)
        #self.fftw_object_b = pyfftw.builders.irfftn(self.in_array_b)

        self.n = n
        ll = np.fix(n/2)
        freqrng = np.arange(-ll,n-ll)
        [omega_x,omega_y,omega_z] = np.meshgrid(freqrng,freqrng,freqrng,indexing='ij')  
        self.omega_x = 2*np.pi*omega_x/n 
        self.omega_y = 2*np.pi*omega_y/n
        self.omega_z = 2*np.pi*omega_z/n   
   

#%% 
def reshift_vol_ref(vol,s,fftw_data=None):
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

    svol = fft.fftshift(fft.ifftn(fft.ifftshift(pim))) 
    if LA.norm(np.imag(svol[:]))/LA.norm(svol[:]) > 1.0e-8:
        raise ValueError("Large imaginary components")
    svol = np.real(svol)
    return svol

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

    if fftw_data is None:
        fftw_data = fftw_data_class(vol)
        
    n = np.size(vol,0)
    if fftw_data.n != n: # Cache is invalid. Recreate.
        fftw_data = fftw_data_class(vol)
        
    phase_x = np.exp(1j*fftw_data.omega_x*s[0])
    phase_y = np.exp(1j*fftw_data.omega_y*s[1])
    phase_z = np.exp(1j*fftw_data.omega_z*s[2])   
    # Force conjugate symmetry. Otherwise this frequency component has no
    # corresponding negative frequency to cancel out its imaginary part.
    if np.mod(n,2) == 0:
         phase_x[0,:,:] = np.real(phase_x[0,:,:])
         phase_y[:,0,:] = np.real(phase_y[:,0,:])
         phase_z[:,:,0] = np.real(phase_z[:,:,0])        
    phases = phase_x*phase_y*phase_z
    
    # n2 = n//2+1
    # vol1 = fft.ifftshift(vol)    
    # pim1 = fftw_data.fftw_object_f(vol1)

    # phases1 = fft.ifftshift(phases)    
    # phases1 = phases1[:,:,:n2]

    # pim = pim1 * phases1
    # svol = fftw_data.fftw_object_b(pim)
    # svol = fft.fftshift(svol)
    
    vol1 = fft.ifftshift(vol)    
    pim1 = pyfftw.interfaces.numpy_fft.fftn(vol1)

    phases1 = fft.ifftshift(phases)    
    # phases1 = phases1[:,:,:n2]

    pim = pim1 * phases1
    svol = pyfftw.interfaces.numpy_fft.ifftn(pim)
    svol = fft.fftshift(svol)
    
    
    
    if LA.norm(np.imag(svol[:]))/LA.norm(svol[:]) > 5.0e-7:
        raise ValueError("Large imaginary components")
    svol = np.real(svol)
   
    # tmpvol = reshift_vol_ref(vol, s)
    # err = np.linalg.norm(tmpvol.ravel()-svol.ravel())/np.linalg.norm(tmpvol.ravel())
    # if err >5.0e-7:
    #     aaa=1
       
    # assert(err<5.0e-7)
    return svol

#%% 
def reshift_vol_int(vol,s):
    # Shift a volume by the vector s.
    # s must be be a vector of integers. 
    # For non integer shifts use reshift_vol

    s = np.array(s)
    if not (np.round(s) == s).all():
        raise ValueError("s must be a vector of integers.")
        
    return np.roll(vol,(-s).astype(int),axis=[0,1,2])
        
    