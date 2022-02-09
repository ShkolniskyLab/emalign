#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 23:42:18 2021

@author: yaelharpaz1
"""

import numpy as np
import math


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

def genRotationsGrid(res):
    # genRotationsGrid generate approximatly equally spaced rotations.
    #   Input:
    #       resolution - the number of samples per 2*pi.
    #                    for example:
    #                        resolution = 50  you get   4484 rotations
    #                        resolution = 75  you get  15236 rotations
    #                        resolution = 100 you get  39365 rotations
    #                        resolution = 150 you get 129835 rotations
    #   Output:
    #       rotations - 3X3Xnumber_of_rotations matrix. of all the rotations.
    # angles - 3Xnumber_of_rotations matrix. each column contains three
    #          angles of the rotation in the following parametrization for
    #          quaternions:
    #          parametrization for SO3
    #               x = sin(tau)* sin(theta)* sin(phi);
    #               y = sin(tau)* sin(theta)* cos(phi);
    #               z = sin(tau)* cos(theta);
    #               w = cos(tau);
    
    counter = 0
    tau_step = (math.pi/2)/(res/4)
    for tau1 in np.arange(tau_step/2, (math.pi/2-tau_step/2), tau_step):
        theta_step = math.pi/(res/2*math.sin(tau1))
        for theta1 in np.arange(theta_step/2, math.pi-theta_step/2, theta_step):
            phi_step = (2*math.pi)/(res*math.sin(tau1)*math.sin(theta1))
            for phi1 in np.arange(0,2*math.pi-phi_step,phi_step):
                counter = counter + 1
    n_of_rotations = counter
    
    angles = np.zeros((3,n_of_rotations))   
    rotations = np.zeros((3,3,n_of_rotations))
    counter = -1
    
    for tau1 in np.arange(tau_step/2, (math.pi/2-tau_step/2), tau_step):
        sintau1 = math.sin(tau1)
        costau1 = math.cos(tau1)
        theta_step = math.pi/(res/2*math.sin(tau1))
        for theta1 in np.arange(theta_step/2, math.pi-theta_step/2, theta_step):
            sintheta1 = math.sin(theta1)
            costheta1 = math.cos(theta1)
            phi_step = (2*math.pi)/(res*math.sin(tau1)*math.sin(theta1))
            for phi1 in np.arange(0,2*math.pi-phi_step,phi_step):
                counter = counter + 1
                angles[:,counter] = np.array([tau1,theta1,phi1])
                rotations[:,:,counter] = q_to_rot(np.array([sintau1*sintheta1*math.sin(phi1), 
                                           sintau1*sintheta1*math.cos(phi1),
                                           sintau1*costheta1,
                                           costau1]).reshape((4,1))).reshape((3,3))
    return rotations    
                                
    
    
    