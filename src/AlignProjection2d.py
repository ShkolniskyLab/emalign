#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:56:14 2021

@author: yaelharpaz1
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:55:46 2021

@author: yaelharpaz1
"""
import numpy as np
import math
import cmath
import logging

from src.cryo_project_itay_finufft import cryo_project
from src.common_finufft import cryo_pft
from src.commonline_R2 import commonline_R2, cryo_normalize
from src.genRotationsGrid import genRotationsGrid
from numpy import linalg as LA


def AlignProjection(projs,vol,verbose=0,opt=None):
    '''
    This function aligns given projection in a given volume.
    This is a secondary algorithm for cryo_align_vols.
    input: 
    projs- projection images for the alignment.
    vol- reference volume.
    verbose- Set verbose to nonzero for verbose printouts (default is zero).
    opt- Structure with optimizer options.
    
    output:
        Rots_est- size=3x3x(size(projs,3)). The estimated rotation matrices of 
                  the projections. If we project the volume in these
                  orientations we will receive the same projections.
        Shifts- size=(size(projs,3))x2. The 2D estimated shift of the 
                projections, first column contained the shift in the x-axis, and  
                the secound colunm in the y-axis.
        corrs- size=size((projs,3))x2. Statistics of the alignment. The i'th  
               entry of the first column contains the correlation of the common    
               lines between the i'th image and all the reference images induced  
               by the best matching rotation. The  i'th entry of the second   
               column contains the mean matching correlation over all tested 
               rotations.
        err_Rots- error calculation between the true rotations and the estimated
                  rotations.
        err_shifts- error calculation between the true shifts and the estimated
                    shifts, in x and y axis.
    Options:
        opt.sym- the symmetry type- 'Cn'\'Dn'\'T'\'O'\'I', where n is the the 
             symmetry order (for example: 'C2'). This input is reqired only for 
             the error calculation.
        opt.Nprojs- number of reference projections for the alignment. (default 
                   is 30).
        opt.isshift- set isshift to nonzero in order to estimate the translations 
                     of the projections (default is zero).
        opt.G- Array of matrices of size 3x3xn containing the symmetry group
               elemnts. This input is for error calculation. 
        opt.trueRots- the true rotations of the projections.
        opt.trueRots_J- the true rotations of the projections in case of 
                      reflection between the projection and the volume.
        opt.trueShifts- the true shifts-(dx,dy) of projs.
        opt.Rots - size=3x3x(size(Rots,3)). a set of candidate rotations.
    '''
    logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger()  
    if verbose == 0 : logger.disabled = True
    
    # Check options:
    if hasattr(opt,'sym'): sym = opt.sym
    else: sym = None    
    
    if hasattr(opt,'Nprojs'): Nprojs = opt.Nprojs
    else: Nprojs = 30
        
    if hasattr(opt,'G'): G = opt.G
    else: G = None
        
    if hasattr(opt,'trueRots'): trueRots = opt.trueRots
    else: trueRots = None
        
    if hasattr(opt,'trueRots_J'): trueRots_J = opt.trueRots_J
    else: trueRots_J = None
        
    if hasattr(opt,'isshift'): isshift = opt.isshift
    else: isshift = 0
        
    if hasattr(opt,'trueShifts'): trueShifts = opt.trueShifts
    else: trueShifts = None  
        
    if hasattr(opt,'Rots'): Rots = opt.Rots
    else: Rots = None  
    
    # Define parameters:
    G_flag = 0
    if sym is not None:   
        s = sym[0] 
        n_s = 0
        if s == 'C' and n_s == 1: 
            G = np.eye(3).reshape((1,3,3))     
    if G is not None:
        G_flag = 1   
        # The symmetry group should be adjusted so it will be acurate for the 
        # projection images. We use the permute function on the projections 
        # such that it replaces the x-axis and the y-axis, so we have to do the
        # same for the symmetry group.
        n_g = np.size(G,0)
        O_g = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]) #x-y axes transformation matrix.
        for i in range(n_g): 
            G[i,:,:] = O_g @ G[i,:,:] @ O_g.T
    refrot = 1
    if trueRots is None: refrot = 0        
    refrot_J = 1
    if trueRots_J is None: refrot_J = 0
    refshift = 1
    if trueShifts is None: refshift = 0
    canrots = 1
    if Rots is None: canrots = 0                
    n = np.size(vol,0); n_r = math.ceil(n/2); L = 360
                        
    # Compute polar Fourier transform of projs:
    logger.info('Computing polar Fourier transform of unaligned projections using n_r= %i, L= %i',n_r,L)  
    projs_hat = cryo_pft(projs,n_r,L)[0]
    # Normalize polar Fourier transforms:
    logger.info('Normalizing the polar Fourier transform of unaligned projections')
    projs_hat = cryo_normalize(projs_hat)
    n_projs = np.size(projs_hat,2)
    
    # Generate candidate rotations and reference projections:
    logger.info('Generating %i reference projections', Nprojs)
    if canrots == 0:
        Rots = genRotationsGrid(75)
    candidate_rots = Rots
    Nrot = np.size(candidate_rots,2)
    logger.info('Using %i candidate rotations for the alignment', Nrot)     
    rots_ref = Rots[:,:,np.random.randint(Nrot, size=Nprojs)] 
    #rots_ref = mat_to_npy('rots_ref_for_AlignProjection2D')
    
    ref_projs = cryo_project(vol, rots_ref)    
    ref_projs = np.transpose(ref_projs,(1,0,2))  
    rots_ref = np.transpose(rots_ref,(1,0,2)) # the true rots
    
    # Compute polar Fourier transform of reference projections:
    logger.info('Computing polar Fourier transform of reference projections using n_r=%i, L=%i', n_r, L)
    refprojs_hat = cryo_pft(ref_projs,n_r,L)[0]
    # Normalize polar Fourier transforms:
    logger.info('Normalizing the polar Fourier transform of reference projections')
    refprojs_hat = cryo_normalize(refprojs_hat)
    
    # Compute the common lines between the candidate rotations and the 
    # references:
    logger.info('Computing the common lines between reference and unaligned projections')     
    Ckj = (-1)*np.ones((Nrot,Nprojs),dtype=int)
    Cjk = (-1)*np.ones((Nrot,Nprojs),dtype=int)
    Mkj = np.zeros((Nrot,Nprojs),dtype=int)
    for k in range(Nrot):
        Rk = np.transpose(candidate_rots[:,:,k])
        for j in range(Nprojs):
            Rj = np.transpose(rots_ref[:,:,j])
            if np.sum(Rk[:,2] @ Rj[:,2]) < 0.999:
                (ckj,cjk) = commonline_R2(Rk,Rj,L)
                # Convert the returned indices ckj and cjk into 1-based
                Ckj[k,j] = ckj
                Cjk[k,j] = cjk
                Mkj[k,j] = 1
    logger.info('Computing the common lines is done')
    
    # Generate shift grid:
    # generating a shift grid on the common lines, and choosing the shift
    # that brings the best correlation in the comparisson between the common 
    # lines. 
    # after applying polar FFT on each projection, the shift in quartesian
    # coordinates- (delta(x),delta(y)) becomes a shift only in the r variable
    # in the common lines (the common line have a specific theta so we have to
    # consider a shift only in the r variable.
    # the equation for the shift phase in the common lines is:
    # exp((-2*pi*i)*r*delta(r)).
    max_s = int(np.round(0.2*np.size(projs_hat,0))) # set the maximum shift.
    s_step = 0.5
    n_shifts = int((2/s_step)*max_s + 1) # always odd number (to have zero value without shift).
    max_r = np.size(projs_hat,0)
    s_vec = np.linspace(-max_s,max_s,n_shifts).reshape((1,n_shifts)) # those are the shifts in the r variable in the common lines. 
    r_vec = np.arange(max_r).reshape((1,max_r))
    s_phases = np.exp((-2*math.pi*cmath.sqrt(-1))/(2*max_r+1)*(r_vec.conj().T @ s_vec)) # size of (n_rXn_shift)
    
    # Main loop- compute the cross correlation: 
    # computing the correlation between the common line, first choose the best
    # shift, and then chose the best rotation.
    logger.info('Aligning unaligned projections using reference projections')
    Rots_est = np.zeros((3,3,n_projs))
    corrs = np.zeros((n_projs,2)) # Statistics on common-lines matching.
    shifts = np.zeros((2,n_projs))
    dtheta = 2*math.pi/L
    if refrot == 1:
        err_Rots = np.zeros((n_projs,1))
    if refshift == 1:
        err_shifts = np.zeros((2,n_projs))
    for projidx in range(n_projs):
        cross_corr_m = np.zeros((Nrot,Nprojs))
        for j in range(Nprojs):
            iidx = np.array(np.where(Mkj[:,j] != 0)).T
            conv_hat = (projs_hat[:,Ckj[iidx,j],projidx].conj() * refprojs_hat[:,Cjk[iidx,j],j]).reshape((n_r,np.size(iidx,axis=0))) # size of (n_rxsize(iidx))
            temp_corr = np.real(s_phases.conj().T @ conv_hat)
            cross_corr_m[iidx,j] = temp_corr.max(axis=0).reshape(iidx.shape)
        # calculating the mean of each row in cross_corr_m:
        cross_corr = (np.sum(cross_corr_m,axis=1)/np.sum(cross_corr_m>0,axis=1)).reshape((Nrot,1))       
        # Find estimated rotation:
        bestRscore = np.amax(cross_corr)
        bestRidx = np.array(np.where(cross_corr == bestRscore))[0,0]
        meanRscore = np.mean(cross_corr)
        corrs[projidx,0] = bestRscore
        corrs[projidx,1] = meanRscore
        Rots_est[:,:,projidx] = candidate_rots[:,:,bestRidx]       
        # Error calculation for estimated rotation:
        if refrot == 1 and G_flag == 1:
            g_est_t = Rots_est[:,:,projidx] @ trueRots[:,:,projidx].T
            n_g = np.size(G,0)
            dist = np.zeros((n_g,1))
            for g_idx in range(n_g):
                dist[g_idx,0] = LA.norm(g_est_t-G[g_idx,:,:],'fro')
            minidx = np.array(np.where(dist == np.amin(dist)))
            g_est = G[minidx[0,0],:,:]           
            R_est = g_est.T @ Rots_est[:,:,projidx]
            R = trueRots[:,:,projidx] @ R_est.T
            err_Rots[projidx,:] = np.rad2deg(math.acos((np.trace(R)-1)/2))
        # Error calculation for reflection case:
        # if there is a reflection between the projection and the volume
        # then, the relation is R_est=gJRJ.
        if refrot_J == 1 and G_flag == 1:
            J3 = np.diag([1, 1, -1])
            g_est_t = Rots_est[:,:,projidx] @ (J3 @ trueRots_J[:,:,projidx] @ J3).T
            n_g = np.size(G,0)
            dist = np.zeros((n_g,1))
            for g_idx in range(n_g):
                dist[g_idx,0] = LA.norm(g_est_t-G[g_idx,:,:],'fro')
            min_idx = np.array(np.where(dist == np.amin(dist)))
            g_est = G[min_idx[0,0],:,:]    
            R_est = J3 @ g_est.T @ Rots_est[:,:,projidx] @ J3
            R = trueRots_J[:,:,projidx] @ R_est.T
            err_2 = np.rad2deg(math.acos((np.trace(R)-1)/2)) 
            if err_Rots[projidx,:] != 0:
                if err_Rots[projidx,:] <= err_2:
                    err_Rots[projidx,:] = err_Rots[projidx,:]
                else:
                    err_Rots[projidx,:] = err_2
            else:
                err_Rots[projidx,:] = err_2 
        
        # Find estimated shift:
        # by least-squares on the estimated rotation with the reference projections. 
        if isshift == 1:
            idx = np.array(np.where(Mkj[bestRidx,:] == 1)).transpose()
            n = np.size(idx,0)
            shift_eq = np.zeros((n,2))
            shift = np.zeros((n,1))
            i=0
            for j in idx:
                conv_hat = projs_hat[:,Ckj[bestRidx,j],projidx].conj() * refprojs_hat[:,Cjk[bestRidx,j],j] # size of (n_rxsize(iidx))
                temp_corr = np.real(s_phases.conj().T @ conv_hat) # size of (n_shiftX1). 
                s_idx = np.array(np.where(temp_corr == temp_corr.max(axis=0)))
                shift[i,0] = s_vec[s_idx[1,0],s_idx[0,0]]
                theta = (Ckj[bestRidx,j]-1)*dtheta # Angle of the common line.
                shift_eq[i,0] = math.sin(theta)
                shift_eq[i,1] = math.cos(theta)
                i = i+1
            shifts[:,projidx] = LA.lstsq(shift_eq,shift)[0].reshape(2)
            # Error calc for estimated shifts:
            if refshift != 0:
                err_shifts[0,projidx] = LA.norm(trueShifts[projidx,0]-shifts[0,projidx],2)
                err_shifts[1,projidx] = LA.norm(trueShifts[projidx,1]-shifts[1,projidx],2)
    if refrot == 1 and G_flag == 1:
        mean_err = np.mean(err_Rots)
        logger.info('Mean error in estimating the rotations of the projections is: %.3f degrees', mean_err)
    if isshift == 1 and refshift == 1:
        mean_err_shift = np.mean(err_shifts)
        logger.info('Mean error in estimating the translations of the projections is: %.3f', mean_err_shift)
        
    logging.shutdown()
    
    if refrot != 0 and G_flag != 0 and isshift != 0 and refshift != 0:
        return Rots_est, shifts, corrs, err_Rots, err_shifts
    elif refrot != 0 and G_flag != 0 and isshift == 0 and refshift == 0:
        return Rots_est, shifts, corrs, err_Rots
    elif refrot == 0 and G_flag == 0 and isshift != 0 and refshift != 0:
        return Rots_est, shifts, corrs, err_shifts
    else:
        return Rots_est, shifts, corrs
    
            
