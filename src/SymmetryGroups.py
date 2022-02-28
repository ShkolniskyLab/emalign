#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:17:56 2021

@author: yaelharpaz1
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy import linalg as LA


def genSymGroup(sym):  
    # This function generates the symmetry group of the given symmetry type. 
    # The symmetry elements are being generated according to the most common 
    # coordinate systems of molecules from the EMDB. Note that it is necessary  
    # to check that the generated symmetry group is indeed the appropriate one.  
    ## Input:
    # sym- the symmetry type- 'Cn'\'Dn'\'T'\'O'\'I', where n is the the symmetry
    #      order.  
    ## Output:
    # G- size=(3,3,N) the symmetry group elemnts, where N is the amount of 
    #    symmetry elements in G.    
    #%%
    s = sym[0] 
    n_s = 0
    if len(sym) > 1:
        n_s = int(sym[1:len(sym)])
    #%% Cyclic symmetry group:
    if s == 'C':
        if n_s == 0:
            raise TypeError("The symmetry order must be an input in case of cyclic symmetry")
        G = np.zeros((n_s,3,3))
        theta = 2*180/n_s
        for i in range(n_s):
            G[i,:,:] = R.from_euler('z', i*theta, degrees=True).as_matrix()    
    #%% Dihedral symmetry group:      
    elif s == 'D':
        if n_s == 0:
            raise TypeError("The symmetry order must be an input in case of dihedral symmetry")         
        G = np.zeros((2*n_s,3,3))
        theta = 2*180/n_s    
        for i in range(n_s):
            G[i,:,:] = R.from_euler('z', i*theta, degrees=True).as_matrix()
        G[n_s,:,:] = R.from_euler('y', 180, degrees=True).as_matrix()
        for i in range(1,n_s):
            G[n_s+i,:,:] = np.matmul(G[n_s,:,:],G[i,:,:])        
    #%% T symmetry group:
    elif s == 'T':    
        n = 12                                  # The size of group T is 12
        G = np.zeros((n,3,3))                       
        G[0,:,:]  = np.eye(3)
        G[1,:,:] = np.array([[0,0, 1],[ 1,0,0],[0, 1,0]])    # axis: [ 1, 1, 1] angle: 120
        G[2,:,:] = np.array([[0, 1,0],[0,0, 1],[ 1,0,0]])    # axis: [ 1, 1, 1] angle: 240    G(2) = G(3)^T
        G[3,:,:] = np.array([[0,0,-1],[ 1,0,0],[0,-1,0]])    # axis: [-1,-1, 1] angle: 120
        G[4,:,:] = np.array([[0, 1,0],[0,0,-1],[-1,0,0]])    # axis: [-1,-1, 1] angle: 240    G(4) = G(3)^T
        G[5,:,:] = np.array([[0,0,-1],[-1,0,0],[0, 1,0]])    # axis: [ 1,-1,-1] angle: 120
        G[6,:,:] = np.array([[0,-1,0],[0,0, 1],[-1,0,0]])    # axis: [ 1,-1,-1] angle: 240    G(6) = G(5)^T
        G[7,:,:] = np.array([[0,0, 1],[-1,0,0],[0,-1,0]])    # axis: [-1, 1,-1] angle: 120
        G[8,:,:] = np.array([[0,-1,0],[0,0,-1],[ 1,0,0]])    # axis: [-1, 1,-1] angle: 240    G(8) = G(7)^T        
        G[9,:,:] = np.array([[ 1,0,0],[0,-1,0],[0,0,-1]])    # axis: [ 1, 0, 0] angle: 180    G(9) = G(9)^T
        G[10,:,:] = np.array([[-1,0,0],[0, 1,0],[0,0,-1]])    # axis: [ 0, 1, 0] angle: 180    G(10) = G(10)^T
        G[11,:,:] = np.array([[-1,0,0],[0,-1,0],[0,0, 1]])    # axis: [ 0, 0, 1] angle: 180    G(11) = G(11)^T
    #%% O symmetry group:
    elif s == 'O':
        n  = 24                                # The size of group O is 24
        G = np.zeros((n,3,3))                        
        G[0,:,:] = np.eye(3) 
        G[1,:,:] = np.array([[0,-1,0],[ 1,0,0],[0,0, 1]])   
        G[2,:,:] = np.array([[0, 1,0],[-1,0,0],[0,0, 1]])      # G_2 = G_1.'
        G[3,:,:] = np.array([[ 1,0,0],[0,0,-1],[0, 1,0]])   
        G[4,:,:] = np.array([[ 1,0,0],[0,0, 1],[0,-1,0]])      # G_4 = G_3.'
        G[5,:,:] = np.array([[0,-1,0],[0,0,-1],[ 1,0,0]])   
        G[6,:,:] = np.array([[0,0, 1],[-1,0,0],[0,-1,0]])      # G_6 = G_5.'
        G[7,:,:] = np.array([[0,-1,0],[0,0, 1],[-1,0,0]])   
        G[8,:,:] = np.array([[0,0,-1],[-1,0,0],[0, 1,0]])      # G_8 = G_7.'
        G[9,:,:] = np.array([[0, 1,0],[0,0, 1],[ 1,0,0]])   
        G[10,:,:] = np.array([[0,0, 1],[ 1,0,0],[0, 1,0]])      # G_10 = G_9.'
        G[11,:,:] = np.array([[0,0, 1],[0, 1,0],[-1,0,0]])   
        G[12,:,:] = np.array([[0,0,-1],[0, 1,0],[ 1,0,0]])      # G_12 = G_11.'
        G[13,:,:] = np.array([[0, 1,0],[0,0,-1],[-1,0,0]])   
        G[14,:,:] = np.array([[0,0,-1],[ 1,0,0],[0,-1,0]])      # G_14 = G_13.'   
        G[15,:,:] = np.array([[-1,0,0],[0,-1,0],[0,0, 1]])      # 2-fold
        G[16,:,:] = np.array([[-1,0,0],[0,0,-1],[0,-1,0]])   
        G[17,:,:] = np.array([[ 1,0,0],[0,-1,0],[0,0,-1]])
        G[18,:,:] = np.array([[0,-1,0],[-1,0,0],[0,0,-1]])   
        G[19,:,:] = np.array([[-1,0,0],[0, 1,0],[0,0,-1]])
        G[20,:,:] = np.array([[0, 1,0],[ 1,0,0],[0,0,-1]])   
        G[21,:,:] = np.array([[-1,0,0],[0,0, 1],[0, 1,0]])
        G[22,:,:] = np.array([[0,0, 1],[0,-1,0],[ 1,0,0]])   
        G[23,:,:] = np.array([[0,0,-1],[0,-1,0],[-1,0,0]])
    #%%
    elif s =='I': 
        n  = 60;                                 # The size of group I is 60
        G = np.zeros((n,3,3))                        
        phi = (1 + np.sqrt(5))/2
        G[0,:,:] = np.eye(3) 
        # 6 rotation axis joining the extreme opposite vertices, by angles 
        # 2pi/5, 4pi/5, 6pi/5 and 8pi/5:
        G[1,:,:] = R.from_rotvec((2*np.pi)/5 * np.array([0, phi,  1]/LA.norm([0, phi,  1]))).as_matrix() 
        G[2,:,:] = R.from_rotvec((4*np.pi)/5 * np.array([0, phi,  1]/LA.norm([0, phi,  1]))).as_matrix()
        G[3,:,:] = R.from_rotvec((6*np.pi)/5 * np.array([0, phi,  1]/LA.norm([0, phi,  1]))).as_matrix()
        G[4,:,:] = R.from_rotvec((8*np.pi)/5 * np.array([0, phi,  1]/LA.norm([0, phi,  1]))).as_matrix()
        
        G[5,:,:] = R.from_rotvec((2*np.pi)/5 * np.array([0, phi, -1]/LA.norm([0, phi, -1]))).as_matrix()
        G[6,:,:] = R.from_rotvec((4*np.pi)/5 * np.array([0, phi, -1]/LA.norm([0, phi, -1]))).as_matrix()
        G[7,:,:] = R.from_rotvec((6*np.pi)/5 * np.array([0, phi, -1]/LA.norm([0, phi, -1]))).as_matrix()
        G[8,:,:] = R.from_rotvec((8*np.pi)/5 * np.array([0, phi, -1]/LA.norm([0, phi, -1]))).as_matrix()
        
        G[9,:,:] = R.from_rotvec((2*np.pi)/5 * np.array([1, 0,  phi]/LA.norm([1, 0,  phi]))).as_matrix()
        G[10,:,:] = R.from_rotvec((4*np.pi)/5 * np.array([1, 0,  phi]/LA.norm([1, 0,  phi]))).as_matrix()
        G[11,:,:] = R.from_rotvec((6*np.pi)/5 * np.array([1, 0,  phi]/LA.norm([1, 0,  phi]))).as_matrix()
        G[12,:,:] = R.from_rotvec((8*np.pi)/5 * np.array([1, 0,  phi]/LA.norm([1, 0,  phi]))).as_matrix()
        
        G[13,:,:] = R.from_rotvec((2*np.pi)/5 * np.array([-1, 0, phi]/LA.norm([-1, 0, phi]))).as_matrix()
        G[14,:,:] = R.from_rotvec((4*np.pi)/5 * np.array([-1, 0, phi]/LA.norm([-1, 0, phi]))).as_matrix()
        G[15,:,:] = R.from_rotvec((6*np.pi)/5 * np.array([-1, 0, phi]/LA.norm([-1, 0, phi]))).as_matrix()
        G[16,:,:] = R.from_rotvec((8*np.pi)/5 * np.array([-1, 0, phi]/LA.norm([-1, 0, phi]))).as_matrix()
        
        G[17,:,:] = R.from_rotvec((2*np.pi)/5 * np.array([phi, -1, 0]/LA.norm([phi, -1, 0]))).as_matrix()
        G[18,:,:] = R.from_rotvec((4*np.pi)/5 * np.array([phi, -1, 0]/LA.norm([phi, -1, 0]))).as_matrix()
        G[19,:,:] = R.from_rotvec((6*np.pi)/5 * np.array([phi, -1, 0]/LA.norm([phi, -1, 0]))).as_matrix()
        G[20,:,:] = R.from_rotvec((8*np.pi)/5 * np.array([phi, -1, 0]/LA.norm([phi, -1, 0]))).as_matrix()
        
        G[21,:,:] = R.from_rotvec((2*np.pi)/5 * np.array([phi,  1, 0]/LA.norm([phi,  1, 0]))).as_matrix()
        G[22,:,:] = R.from_rotvec((4*np.pi)/5 * np.array([phi,  1, 0]/LA.norm([phi,  1, 0]))).as_matrix()
        G[23,:,:] = R.from_rotvec((6*np.pi)/5 * np.array([phi,  1, 0]/LA.norm([phi,  1, 0]))).as_matrix()
        G[24,:,:] = R.from_rotvec((8*np.pi)/5 * np.array([phi,  1, 0]/LA.norm([phi,  1, 0]))).as_matrix()
    
        # 10 rotation axis joining the centers of opposite faces, by angles 
        # 2pi/3 and 4pi/3:  
        G[25,:,:] = R.from_rotvec((2*np.pi)/3 * np.array([0,-1,phi**2]/LA.norm([0,-1,phi**2]))).as_matrix()
        G[26,:,:] = R.from_rotvec((4*np.pi)/3 * np.array([0,-1,phi**2]/LA.norm([0,-1,phi**2]))).as_matrix()
        
        G[27,:,:] = R.from_rotvec((2*np.pi)/3 * np.array([0, 1,phi**2]/LA.norm([0, 1,phi**2]))).as_matrix()
        G[28,:,:] = R.from_rotvec((4*np.pi)/3 * np.array([0, 1,phi**2]/LA.norm([0, 1,phi**2]))).as_matrix()
        
        G[29,:,:] = R.from_rotvec((2*np.pi)/3 * np.array([-1,phi**2,0]/LA.norm([-1,phi**2,0]))).as_matrix()
        G[30,:,:] = R.from_rotvec((4*np.pi)/3 * np.array([-1,phi**2,0]/LA.norm([-1,phi**2,0]))).as_matrix()
        
        G[31:,:] = R.from_rotvec((2*np.pi)/3 * np.array([ 1,phi**2,0]/LA.norm([ 1,phi**2,0]))).as_matrix()
        G[32,:,:] = R.from_rotvec((4*np.pi)/3 * np.array([ 1,phi**2,0]/LA.norm([ 1,phi**2,0]))).as_matrix()
        
        G[33,:,:] = R.from_rotvec((2*np.pi)/3 * np.array([phi**2,0,-1]/LA.norm([phi**2,0,-1]))).as_matrix()
        G[34,:,:] = R.from_rotvec((4*np.pi)/3 * np.array([phi**2,0,-1]/LA.norm([phi**2,0,-1]))).as_matrix()
        
        G[35,:,:] = R.from_rotvec((2*np.pi)/3 * np.array([phi**2,0, 1]/LA.norm([phi**2,0, 1]))).as_matrix()
        G[36,:,:] = R.from_rotvec((4*np.pi)/3 * np.array([phi**2,0, 1]/LA.norm([phi**2,0, 1]))).as_matrix()
        
        G[37,:,:] = R.from_rotvec((2*np.pi)/3 * np.array([  1, -1, 1]/LA.norm([  1, -1, 1]))).as_matrix()
        G[38,:,:] = R.from_rotvec((4*np.pi)/3 * np.array([  1, -1, 1]/LA.norm([  1, -1, 1]))).as_matrix()
    
        G[39,:,:] = R.from_rotvec((2*np.pi)/3 * np.array([ -1,  1, 1]/LA.norm([ -1,  1, 1]))).as_matrix()
        G[40,:,:] = R.from_rotvec((4*np.pi)/3 * np.array([ -1,  1, 1]/LA.norm([ -1,  1, 1]))).as_matrix()

        G[41,:,:] = R.from_rotvec((2*np.pi)/3 * np.array([  1, 1, -1]/LA.norm([  1, 1, -1]))).as_matrix()
        G[42,:,:] = R.from_rotvec((4*np.pi)/3 * np.array([  1, 1, -1]/LA.norm([  1, 1, -1]))).as_matrix()

        G[43,:,:] = R.from_rotvec((2*np.pi)/3 * np.array([  1, 1,  1]/LA.norm([  1, 1,  1]))).as_matrix()
        G[44,:,:] = R.from_rotvec((4*np.pi)/3 * np.array([  1, 1,  1]/LA.norm([  1, 1,  1]))).as_matrix()
    
        # 15 rotation axis joining the midpoints of opposite edges, by angle pi:
        G[45,:,:] = R.from_rotvec(np.pi * np.array([  0,   1,  0]/LA.norm([  0,   1,  0]))).as_matrix()
        G[46,:,:] = R.from_rotvec(np.pi * np.array([  0,   0,  1]/LA.norm([  0,   0,  1]))).as_matrix()                                   
        G[47,:,:] = R.from_rotvec(np.pi * np.array([  1,   0,  0]/LA.norm([  1,   0,  0]))).as_matrix()                                  
        G[48,:,:] = R.from_rotvec(np.pi * np.array([-1/phi,1,phi]/LA.norm([-1/phi,1,phi]))).as_matrix()
        G[49,:,:] = R.from_rotvec(np.pi * np.array([1/phi,-1,phi]/LA.norm([1/phi,-1,phi]))).as_matrix()
        G[50,:,:] = R.from_rotvec(np.pi * np.array([1/phi,1,-phi]/LA.norm([1/phi,1,-phi]))).as_matrix()
        G[51,:,:] = R.from_rotvec(np.pi * np.array([1/phi, 1,phi]/LA.norm([1/phi, 1,phi]))).as_matrix()
        G[52,:,:] = R.from_rotvec(np.pi * np.array([-1,phi,1/phi]/LA.norm([-1,phi,1/phi]))).as_matrix()
        G[53,:,:] = R.from_rotvec(np.pi * np.array([1,-phi,1/phi]/LA.norm([1,-phi,1/phi]))).as_matrix()
        G[54,:,:] = R.from_rotvec(np.pi * np.array([1,phi,-1/phi]/LA.norm([1,phi,-1/phi]))).as_matrix()
        G[55,:,:] = R.from_rotvec(np.pi * np.array([1,phi, 1/phi]/LA.norm([1,phi, 1/phi]))).as_matrix()
        G[56,:,:] = R.from_rotvec(np.pi * np.array([-phi,1/phi,1]/LA.norm([-phi,1/phi,1]))).as_matrix()
        G[57,:,:] = R.from_rotvec(np.pi * np.array([phi,-1/phi,1]/LA.norm([phi,-1/phi,1]))).as_matrix()
        G[58,:,:] = R.from_rotvec(np.pi * np.array([phi,1/phi,-1]/LA.norm([phi,1/phi,-1]))).as_matrix()
        G[59,:,:] = R.from_rotvec(np.pi * np.array([phi,1/phi, 1]/LA.norm([phi,1/phi, 1]))).as_matrix()
        
        O_g = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]) #x-y axes transformation matrix.
        for i in range(60): 
            G[i,:,:] = O_g @ G[i,:,:] @ O_g.T
    
    else: 
        raise TypeError("sym was not entered properly")               
    return G


