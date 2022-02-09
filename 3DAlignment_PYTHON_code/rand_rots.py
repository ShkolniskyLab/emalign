#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 18:46:35 2021

@author: yaelharpaz1
"""

import numpy as np

def qrand(k):
    # Generate K random uniformly distributed quaternions.
    # Each quaternions is a four-elements column vector. Returns a matrix of
    # size 4xK.
    
    # The 3-sphere S^3 in R^4 is a double cover of the rotation group SO(3),
    # SO(3) = RP^3. 
    # We identify unit norm quaternions a^2+b^2+c^2+d^2=1 with group elements. 
    # The antipodal points (-a,-b,-c,-d) and (a,b,c,d) are identified as the
    # same group elements, so we take a>=0.
    from scipy.stats import norm
    q = norm.ppf(np.random.rand(k, 4).T)
    
    l2_norm = np.sqrt(q[0,:]**2 + q[1,:]**2 + q[2,:]**2 +q[3,:]**2)
    for i in range(4):
        q[i,:] = q[i,:]/l2_norm
    for k in range(k):
        if q[0,k] < 0:
            q[:,k] = -q[:,k]
            
    return q

def q_to_rot(q):
    # Convert a quaternion into a rotation matrix.
    
    #   Input: 
    #           q: quaternion. May be a vector of dimensions 4 x n
    #   Output: 
    #           rot_matrix: 3x3xn rotation matrix
    
    #   Yariv Aizenbud 31.01.2016
    
    n = np.size(q, axis=1)
    rot_matrix = np.zeros((3,3,n))
    
    rot_matrix[0,0,:] = q[0,:]**2 + q[1,:]**2 - q[2,:]**2 - q[3,:]**2
    rot_matrix[0,1,:] = 2*q[1,:]*q[2,:] - 2*q[0,:]*q[3,:] 
    rot_matrix[0,2,:] = 2*q[0,:]*q[2,:] + 2*q[1,:]*q[3,:] 
    
    rot_matrix[1,0,:] = 2*q[1,:]*q[2,:] + 2*q[0,:]*q[3,:]
    rot_matrix[1,1,:] = q[0,:]**2 - q[1,:]**2 + q[2,:]**2 - q[3,:]**2 
    rot_matrix[1,2,:] = -2*q[0,:]*q[1,:] + 2*q[2,:]*q[3,:]
    
    rot_matrix[2,0,:] = -2*q[0,:]*q[2,:] + 2*q[1,:]*q[3,:] 
    rot_matrix[2,1,:] = 2*q[0,:]*q[1,:] + 2*q[2,:]*q[3,:] 
    rot_matrix[2,2,:] = q[0,:]**2 - q[1,:]**2 - q[2,:]**2 + q[3,:]**2
    
    return rot_matrix

def rand_rots(n):
    # rand_rots generate random rotations
    # Input
    #    n: The number of rotations to generate.
    # Output
    #    rot_matrices: An array of size 3-by-3-by-n containing n rotation matrices
    #    sampled from the unifoorm distribution on SO(3).

    qs = qrand(n)
    rot_matrices = q_to_rot(qs)
    
    return rot_matrices
