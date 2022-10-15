#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 11:11:19 2021

@author: yaelharpaz1
"""

import numpy as np
import math
import cmath
import scipy.spatial.transform
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
        
        self.in_array_f_0 = pyfftw.empty_aligned((n,n), dtype=real_type)
        self.out_array_f_0 = pyfftw.empty_aligned((n2,n), dtype=complex_type)
        self.in_array_f_1 = pyfftw.empty_aligned((n,n), dtype=real_type)
        self.out_array_f_1 = pyfftw.empty_aligned((n,n2), dtype=complex_type)
        self.in_array_b_0 = pyfftw.empty_aligned((n2,n), dtype=complex_type)
        self.out_array_b_0 = pyfftw.empty_aligned((n,n), dtype=real_type)
        self.in_array_b_1 = pyfftw.empty_aligned((n,n2), dtype=complex_type)
        self.out_array_b_1 = pyfftw.empty_aligned((n,n), dtype=real_type)
        
        self.fftw_object_0 = pyfftw.FFTW(self.in_array_f_0,
                          self.out_array_f_0,
                          direction="FFTW_FORWARD",
                          flags=("FFTW_ESTIMATE", ),
                          axes=(0,),
                          threads=num_threads)

        self.fftw_object_1 = pyfftw.FFTW(self.in_array_f_1,
                          self.out_array_f_1,
                          direction="FFTW_FORWARD",
                          flags=("FFTW_ESTIMATE", ),
                          axes=(1,),
                          threads=num_threads)

        
        self.ifftw_object_0 = pyfftw.FFTW(self.in_array_b_0,
                          self.out_array_b_0,
                          direction="FFTW_BACKWARD",
                          flags=("FFTW_ESTIMATE", ),
                          axes=(0,),
                          normalise_idft=True,
                          threads=num_threads)
        
        self.ifftw_object_1 = pyfftw.FFTW(self.in_array_b_1,
                          self.out_array_b_1,
                          direction="FFTW_BACKWARD",
                          flags=("FFTW_ESTIMATE", ),
                          axes=(1,),
                          normalise_idft=True,
                          threads=num_threads)


#%%
def fastrotate3d(vol,Rot,fftw_data=None):
    #FASTROTATE3D Rotate a 3D volume by a given rotation matrix.
    # Input parameters:
    #  INPUT    Volume to rotate, can be odd or even. 
    #  Rot        3x3 rotation matrix.
    # Output parameters:
    #  OUTPUT   The rotated volume.
    # Examples:
    #   Rot=rand_rots(1);
    #   rvol=fastrotate3d(vol,Rot);
    #Yoel Shkolnisky, November 2013.
    
    if fftw_data is None:
        fftw_data = fftw_data_class(vol)

    Rot_obj = scipy.spatial.transform.Rotation.from_matrix(Rot)
    [psi,theta,phi]  = Rot_obj.as_euler('xyz')

    psid = psi*180/np.pi
    thetad = theta*180/np.pi 
    phid = phi*180/np.pi
    
    tmp = fastrotate3x(vol,psid,fftw_data)
    tmp = fastrotate3y(tmp,thetad,fftw_data)
    vol_out = fastrotate3z(tmp,phid,fftw_data)
    return vol_out

#%%
def adjustrotate(phi):
    # Decompose a rotation CCW by phi into a rotation of mult90 times 90
    # degrees followed by a rotation by phi2, where phi2 is between -45 and 45.
    # mult90 is an integer between 0 and 3 describing by how many multiples of
    # 90 degrees the image should be rotated so that an additional rotation by
    # phi2 is equivalent to rotation by phi.
    
    phi = np.mod(phi,360)
    mult90 = 0
    phi2 = phi   
    # Note that any two consecutive cases can be combine, but I decided to
    # leave them separated for clarity.
    if phi >= 45 and phi < 90: mult90 = 1; phi2 = -(90-phi)
    elif phi >= 90 and phi < 135: mult90 = 1; phi2 = phi-90     
    elif phi >= 135 and phi < 180: mult90 = 2; phi2 = -(180-phi)
    elif phi >= 180 and phi < 225: mult90 = 2; phi2 = phi-180
    elif phi >= 215 and phi < 270: mult90 = 3; phi2 = -(270-phi)
    elif phi >= 270 and phi < 315: mult90 = 3; phi2 = phi-270
    elif phi >= 315 and phi < 360: mult90 = 0; phi2 = phi-360
    return phi2, mult90    
    
#%%  
def fastrotateprecomp(SzX,SzY,phi):
    # Compute the interpolation tables required to rotate an image with SzX
    # rows and SzY columns by an angle phi CCW.
    #
    # This function is used to accelerate fastrotate, in case many images are
    # needed to be rotated by the same angle. In such a case it allows to
    # precompute the interpolation tables only once instead of computing them
    # for each image.
    #
    # M is a structure containing phi, Mx, My, where Mx and My are the
    # interpolation tables used by fastrotate.
    
    # Adjust the rotation angle to be between -45 and 45 degrees.
    [phi,mult90] = adjustrotate(phi)   
    phi = np.pi*phi/180
    phi = -phi # To match Yaroslavsky's code which rotates CW.    
    if np.mod(SzY,2) == 0:
        cy = SzY/2+1
        sy = 1/2 # By how much should we shift the cy to get the center of the image
    else:
        cy = (SzY+1)/2
        sy = 0    
    if np.mod(SzX,2) == 0:
        cx = SzX/2+1 # By how much should we shift the cy to get the center of the image
        sx = 1/2
    else: cx = (SzX+1)/2; sx = 0
    # Precompte My and Mx:
    My = np.zeros((SzY,SzX)).astype(complex)
    r = np.arange(0,cy).astype(int)
    u = (1-math.cos(phi))/math.sin(phi+2.2204e-16)
    alpha1 = 2*np.pi*cmath.sqrt(-1)*(r)/SzY
    for x in range(SzX):
        Ux = u*(x+1-cx+sx)
        My[r,x] = np.exp(alpha1*Ux)
        My[np.arange(SzY-1,cy-1,-1).astype(int),x] = np.conj(My[np.arange(1,cy-2*sy).astype(int),x])
    My = My.T # Remove when implementing using the loops below.  
    Mx = np.zeros((SzX,SzY)).astype(complex)
    r = np.arange(0,cx).astype(int)
    u = -math.sin(phi)
    alpha2 = 2*np.pi*cmath.sqrt(-1)*(r)/SzX
    for y in range(SzY):
        Uy = u*(y+1-cy+sy)
        Mx[r,y] = np.exp(alpha2*Uy)
        Mx[np.arange(SzX-1,cx-1,-1).astype(int),y] = np.conj(Mx[np.arange(1,cx-2*sx).astype(int),y])
    class Struct:
        pass
    M = Struct() 
    M.phi = phi
    M.Mx = Mx
    M.My = My
    M.mult90 = mult90
    return M

#%%
def fastrotate(im,phi,M=None, fftw_data=None):
    # 3-step image rotation by shearing.
    # 
    # input parameters:
    #  im      Image to rotate, can be odd or even. 2D array of size nxn.
    #  phi      Rotation angle in degrees CCW. Can be any angle (not limited
    #           like fastrotate_ref). Note that Yaroslavsky's code take phi CW.
    #  M        (Optional) Precomputed interpolation tables, as generated by
    #           fastrotateprecomp. If M is given than phi is ignored. This is
    #           useful if many images need to be rotated by the same angle,
    #           since then the computation of the same interpolation tables
    #           over and over again is avoided.
    # Output parameters:
    #  im_out   The rotated image.
    
    if fftw_data is None:
        fftw_data = fftw_data_class(im)
    
    im = im.copy()  # Create a copy of the input the prevent changing the 
                      # calling object
    SzX, SzY = im.shape

    if M is None:
        M = fastrotateprecomp(SzX,SzY,phi)
    Mx = M.Mx
    My = M.My 
    mult90 = M.mult90
    im_out = np.zeros((SzX,SzY))

    n = im.shape[0]
    n2 = n//2 + 1
    spinput_0 = np.zeros((n2,n),dtype=np.complex128)
    spinput_1 = np.zeros((n,n2),dtype=np.complex128)
    
    
    # Rotate by multiples of 90 degrees.
    if mult90 == 1: im = rot90(im)  
    elif mult90 == 2: im = rot180(im)
    elif mult90 == 3: im = rot270(im)
    elif mult90 != 0: TypeError('Invalid value for mult90')
        
    # Old code:
    #spinput = fft(vol[:,:,k],n=None,axis=1)
    #spinput = spinput*My
    #vol_out[:,:,k] = np.real(ifft(spinput,n=None,axis=1))

    fftw_data.in_array_f_1[:,:] = im
    spinput_1[:,:] = fftw_data.fftw_object_1(fftw_data.in_array_f_1)
    spinput_1 = spinput_1*My[:,0:n2]
    fftw_data.in_array_b_1[:,:] = spinput_1[:,:]
    im_out[:,:] = fftw_data.ifftw_object_1(fftw_data.in_array_b_1)
        

    # Old code:
    #spinput = fft(vol_out[:,:,k],n=None,axis=0)
    #spinput = spinput*Mx
    #vol_out[:,:,k] = np.real(ifft(spinput,n=None,axis=0))

    fftw_data.in_array_f_0[:,:] = im_out
    spinput_0[:,:] = fftw_data.fftw_object_0(fftw_data.in_array_f_0)
    spinput_0 = spinput_0*Mx[0:n2,:]
    fftw_data.in_array_b_0[:,:] = spinput_0[:,:]
    im_out[:,:] = fftw_data.ifftw_object_0(fftw_data.in_array_b_0)
        
                
    # Old code:
    #spinput = fft(vol_out[:,:,k],n=None,axis=1)
    #spinput = spinput*My
    #vol_out[:,:,k] = np.real(ifft(spinput,n=None,axis=1))
    
    fftw_data.in_array_f_1[:,:] = im_out
    spinput_1[:,:] = fftw_data.fftw_object_1(fftw_data.in_array_f_1)
    spinput_1 = spinput_1*My[:,0:n2]
    fftw_data.in_array_b_1[:,:] = spinput_1[:,:]
    im_out[:,:] = fftw_data.ifftw_object_1(fftw_data.in_array_b_1)
        
    return im_out 
   
#%%
def rot90(A):
    # Rotate the image A by 90 degrees CCW.
    #   B = rot90(A)
    B = A.T
    B = np.flip(B,0)
    return B

#%%
def rot180(A):
    # Rotate the image A by 180 degrees CCW.
    #   B = rot180(A) 
    B = np.flip(A,0)
    B = np.flip(B,1)
    return B

#%% 
def rot270(A):
    # Rotate the image A by 270 degrees CCW.
    #   B = rot270(A)
    
    B = A.T
    B = np.flip(B,1)
    return B

#%%
def fastrotate3x(vol,phi,fftw_data=None):
    #FASTROTATE3X Rotate a 3D volume around the x-axis.
    # Input parameters:
    #  INPUT    Volume to rotate, can be odd or even. 
    #  phi      Rotation angle in degrees CCW. 
    #  M        (Optional) Precomputed interpolation tables, as generated by
    #           fastrotateprecomp. If M is given than phi is ignored. 
    #
    # Output parameters:
    #  OUTPUT   The rotated volume.
    #
    # Examples:
    #
    #   rvol=fastrotate3x(vol,20);
    #
    #   M=fastrotateprecomp(size(vol,2),size(vol,3),20);
    #   rvol=fastrotate(vol,[],M);

    if fftw_data is None:
        fftw_data = fftw_data_class(vol)

    SzX = np.size(vol,0); SzY = np.size(vol,1); SzZ = np.size(vol,2)  
    # Precompte M
    M = fastrotateprecomp(SzY,SzZ,phi)    
    vol_out = np.zeros((SzX,SzY,SzZ),dtype=float)
    for k in range(SzX):
        im = vol[:,k,:]
        rim = fastrotate(im,[],M,fftw_data)
        vol_out[:,k,:] = rim
    return vol_out

#%%
def fastrotate3y(vol,phi,fftw_data=None):
    #FASTROTATE3X Rotate a 3D volume around the x-axis.
    # Input parameters:
    #  INPUT    Volume to rotate, can be odd or even. 
    #  phi      Rotation angle in degrees CCW. 
    #  M        (Optional) Precomputed interpolation tables, as generated by
    #           fastrotateprecomp. If M is given than phi is ignored. 
    #
    # Output parameters:
    #  OUTPUT   The rotated volume.
    #
    # Examples:
    #
    #   rvol=fastrotate3x(vol,20);
    #
    #   M=fastrotateprecomp(size(vol,2),size(vol,3),20);
    #   rvol=fastrotate(vol,[],M);

    if fftw_data is None:
        fftw_data = fftw_data_class(vol)

    SzX = np.size(vol,0); SzY = np.size(vol,1); SzZ = np.size(vol,2)
    # Precompte M
    M = fastrotateprecomp(SzX,SzY,-phi)
    vol_out = np.zeros((SzX,SzY,SzZ),dtype=float)
    for k in range(SzY):
        im = vol[k,:,:]
        rim = fastrotate(im,[],M,fftw_data)
        vol_out[k,:,:] = rim
    return vol_out

#%%
def fastrotate3z(vol,phi,fftw_data=None):
    #FASTROTATE3X Rotate a 3D volume around the x-axis.
    # Input parameters:
    #  INPUT    Volume to rotate, can be odd or even. 
    #  phi      Rotation angle in degrees CCW. 
    #  M        (Optional) Precomputed interpolation tables, as generated by
    #           fastrotateprecomp. If M is given than phi is ignored. 
    #
    # Output parameters:
    #  OUTPUT   The rotated volume.
    #
    # Examples:
    #
    #   rvol=fastrotate3x(vol,20);
    #
    #   M=fastrotateprecomp(size(vol,2),size(vol,3),20);
    #   rvol=fastrotate(vol,[],M);
    
    if fftw_data is None:
        fftw_data = fftw_data_class(vol)
    
    SzX = np.size(vol,0); SzY = np.size(vol,1); SzZ = np.size(vol,2)    
    # Precompte M
    M = fastrotateprecomp(SzX,SzY,-phi)    
    vol_out = np.zeros((SzX,SzY,SzZ),dtype=float)
    for k in range(SzZ):
        im = vol[:,:,k]
        rim = fastrotate(im,[],M,fftw_data)
        vol_out[:,:,k] = rim
    return vol_out

#import scipy.io as matio
#mat_vars = matio.loadmat("../mattemp.mat")
#vol = mat_vars["vol"]
#R = mat_vars["R"]
#vol_rot = fastrotate3d(vol, R)
#print(np.linalg.norm(vol_rot - mat_vars["vol_rot"])/np.linalg.norm(vol_rot))
#vol_rot = fastrotate3d(vol, R)
#print(np.linalg.norm(vol_rot - mat_vars["vol_rot"])/np.linalg.norm(vol_rot))
#vol2 = fastrotate3d(vol_rot,R.transpose())
#np.corrcoef(vol2.ravel(),vol.ravel())
#aaa = 1
