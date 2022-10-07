#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 20:06:16 2021

@author: yaelharpaz1
"""
import numpy as np
import math
from numpy import linalg as LA
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from src.common_finufft import cryo_downsample
from src.cryo_project_itay_finufft import cryo_project
from src.genRotationsGrid import genRotationsGrid
from src.AlignProjection2d import AlignProjection
from src.fastrotate3d import fastrotate3d
from src.register_translations_3d import register_translations_3d
from src.register_translations_3d import refine3DshiftBFGS
from src.reshift_vol import reshift_vol
from src.SymmetryGroups import genSymGroup
import logging


def fastAlignment3D(sym, vol1, vol2, n, Nprojs=30, trueR=None, G_group=None, refrot=0, verbose=0):
    '''
    This function does the work for AlignVolumes.
    Input:
    sym- the symmetry type- 'Cn'\'Dn'\'T'\'O'\'I', where n is the the
         symmetry order (for example: 'C2').
    vol1- 3D reference volume that vol2 should be aligned accordingly.
    vol2- 3D volume to be aligned.
    verbose- Set verbose to nonzero for verbose printouts (default is zero).
    n- the size of vol1 and vol2.
    Nprojs- number of reference projections for the alignment.
    trueR- the true rotation matrix between vol2 and vol1.
    refrot- indicator for true_R. If true_R exist then refrot=1, else
            refrot=0.
    G_group- size=(n,3,3) all n symmetry group elemnts.

    output:
    Rest- the estimated rotation between vol_2 and vol_1 without reflection.
    Rest_J- the estimated rotation between vol_2 and vol_1 with reflection.
    '''
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger()
    if verbose == 0: logger.disabled = True
    # Generate reference projections from vol2:
    logger.info('Generating %i reference projections.', Nprojs)
    Rots = genRotationsGrid(75)
    sz_Rots = np.size(Rots, 2)
    R_ref = Rots[:, :, np.random.randint(sz_Rots, size=Nprojs)]  # size (3,3,N_projs)
    # R_ref = mat_to_npy('R_ref_for_fastAlignment3D')

    ref_projs = cryo_project(vol2, R_ref)
    ref_projs = np.transpose(ref_projs, (1, 0, 2))
    R_ref = np.transpose(R_ref, (1, 0, 2))  # the true rotations.

    # Align reference projections to vol1:
    class Struct:
        pass

    opt = Struct()
    opt.Nprojs = Nprojs;
    opt.G = G_group;
    opt.Rots = Rots;
    opt.sym = sym;
    logger.info('Aligning reference projections of vol2 to vol1.')
    if refrot == 1:
        R = trueR;
        R = R.T;
        R = R[:, [1, 0, 2]][[1, 0, 2]];
        trueR_tild = np.zeros((3, 3, Nprojs))
        trueR_tild_J = np.zeros((3, 3, Nprojs))
        for i in range(Nprojs):
            trueR_tild[:, :, i] = R @ R_ref[:, :, i]
            J3 = np.diag([1, 1, -1])
            trueR_tild_J[:, :, i] = J3 @ R @ J3 @ R_ref[:, :, i]
        opt.trueRots = trueR_tild;
        opt.trueRots_J = trueR_tild_J;
        R_tild = AlignProjection(ref_projs, vol1, verbose, opt)  # size=(3,3,N_projs).
    else:
        R_tild = AlignProjection(ref_projs, vol1, verbose, opt)  # size (3,3,N_projs).
    # Synchronization:
    # A synchronization algorithm is used In order to revel the symmetry
    # elements of the reference projections. The relation between the volumes
    # is V2(r)=V1(Or). Denote the rotation between the volumes as X.
    # 1. In the case there is no reflection between the volumes, the rotation
    #    Ri_tilde estimates giORi, therefore the approximation is O =
    #    gi.'Ri_tildeRi.', where g_i is the symmetry group element of reference
    #    image i. If we define Xi=Ri*Ri_tilde.' then we get Xi.'*Xj=g_i*g_j.'.
    # 2. In the case there is a reflection between the volumes, the rotation
    #    Ri_tilde estimates qiJXRiJ, where O=JX. We have that qiJ=Jqi_tilde,
    #    therefore the approximation is X=qi_tilde.'JRi_tildeJRi.', where
    #    qi_tilde is a symmetry element in the symmetry group of
    #    V1_tilde(r)=V1(Jr). If we define  Xi=Ri*(J*Ri_tild*J).', then we also
    #    get Xi.'*Xj=qi_tilde*qj_tilde.'.
    # Therefore, we can construct the synchronization matrix Xij=Xi.'*Xj for
    # both cases. Then, estimate the group elemnts for each image with and
    # whitout reflection, and latter choose the option that best describes the
    # relation between the two volumes.

    # Estimate X with or without reflection:
    R_tild = R_tild[0]
    X_mat = np.zeros((3, 3, Nprojs))
    X_mat_J = np.zeros((3, 3, Nprojs))
    J3 = np.diag([1, 1, -1])
    for i in range(Nprojs):
        X_mat[:, :, i] = R_ref[:, :, i] @ R_tild[:, :, i].T
        X_mat_J[:, :, i] = R_ref[:, :, i] @ (J3 @ R_tild[:, :, i] @ J3).T
    # Construct the synchronization matrix with and without reflection:
    X_ij = np.zeros((3 * Nprojs, 3 * Nprojs))
    X_ij_J = np.zeros((3 * Nprojs, 3 * Nprojs))
    for i in range(Nprojs):
        for j in range(i + 1, Nprojs):
            X_ij[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)] = X_mat[:, :, i].T @ X_mat[:, :, j]
            X_ij_J[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)] = X_mat_J[:, :, i].T @ X_mat_J[:, :, j]
    # Enforce symmetry:
    X_ij = X_ij + X_ij.T
    X_ij = X_ij + np.eye(np.size(X_ij, 0))
    X_ij_J = X_ij_J + X_ij_J.T
    X_ij_J = X_ij_J + np.eye(np.size(X_ij_J, 0))
    # Define v=[g_1.',..., g_N_projs.'].' (v is of size 3*N_projx3), then
    # X_ij=v*v.', and Xij*v=N_projs*v. Thus, v is an eigenvector of Xij. The
    # matrix Xij should be of rank 3. find the top 3 eigenvectors:
    # without reflection:
    s, U = LA.eigh(X_ij)  # s = np.diag(s);
    ii = np.argsort(s, axis=0)[::-1]  # s = np.sort(s,axis=0)[::-1]
    U = U[:, ii]
    V = U[:, 0:3]
    # With reflection:
    sJ, UJ = LA.eigh(X_ij_J)  # sJ = np.diag(sJ);
    iiJ = np.argsort(sJ, axis=0)[::-1]  # sJ = np.sort(sJ,axis=0)[::-1];
    UJ = UJ[:, iiJ]
    VJ = UJ[:, 0:3]
    # estimating G:
    # Estimate the group elemnts for each reference image. G denotes the
    # estimated group without reflection, and G_J with reflection. This
    # estimation is being done from the eigenvector v by using a rounding
    # algorithm over SO(3) for each 3x3 block of v.
    G = np.zeros((Nprojs, 3, 3))
    G_J = np.zeros((Nprojs, 3, 3))
    for i in range(Nprojs):
        B = V[3 * i:3 * (i + 1), :]
        u_tmp, s_tmp, v_tmp = LA.svd(B)
        B_round = LA.det(u_tmp @ v_tmp) * (u_tmp @ v_tmp)
        G[i, :, :] = B_round.T
        # reflected case:
        BJ = VJ[3 * i:3 * (i + 1), :]
        uJ_tmp, sJ_tmp, vJ_tmp = LA.svd(BJ)
        BJ_round = LA.det(uJ_tmp @ vJ_tmp) * (uJ_tmp @ vJ_tmp)
        G_J[i, :, :] = BJ_round.T
    # Set the global rotation to be an element from the symmetry group:
    # The global rotation from the synchronization can be any rotation matrix
    # from SO(3). So, in order to get the estimated symmetry elements to be
    # from the symmetry group we set the global rotation to be also an element
    # from the symmetry group.
    O1 = G[0, :, :].T
    O1_J = G_J[0, :, :].T
    G_est = np.zeros((Nprojs, 3, 3))
    G_J_est = np.zeros((Nprojs, 3, 3))
    for i in range(Nprojs):
        G_est[i, :, :] = O1 @ G[i, :, :]
        G_J_est[i, :, :] = O1_J @ G_J[i, :, :]
        # Estimating the rotation:
    # Estimate the two candidate orthogonal transformations.
    for i in range(Nprojs):
        X_mat[:, :, i] = X_mat[:, :, i] @ G_est[i, :, :].T
        X_mat_J[:, :, i] = X_mat_J[:, :, i] @ G_J_est[i, :, :].T
    X = np.mean(X_mat, axis=2)
    X_J = np.mean(X_mat_J, axis=2)
    # Without reflection:
    R = X
    U, S, V = LA.svd(R)  # Project R to the nearest rotation.
    R_est = U @ V
    assert LA.det(R_est) > 0
    R_est = R_est[:, [1, 0, 2]][[1, 0, 2]]
    R_est = R_est.T
    # reflected case:
    R_J = X_J
    U, S, V = LA.svd(R_J)  # Project R to the nearest rotation.
    R_est_J = U @ V
    assert LA.det(R_est_J) > 0
    R_est_J = R_est_J[:, [1, 0, 2]][[1, 0, 2]]
    R_est_J = R_est_J.T

    logging.shutdown()

    return R_est, R_est_J


# %%
def eval3Dmatchaux(X, vol1, vol2):
    psi = X[0]
    theta = X[1]
    phi = X[2]
    dx = X[3]
    dy = X[4]
    dz = X[5]
    r = R.from_euler('xyz', [psi, theta, phi], degrees=False)
    Rot = r.as_matrix()

    vol2_r = fastrotate3d(vol2, Rot)

    vol2_rs = reshift_vol(vol2_r, np.array([dx, dy, dz]))
    c = np.mean(np.corrcoef(vol1.ravel(), vol2_rs.ravel(), rowvar=False)[0, 1:]).astype('float64')
    e = (1 - c).astype('float64')
    return e


# %%
def refine3DmatchBFGS(vol1, vol2, R1, estdx):
    # Create initial guess vector
    R1 = R.from_matrix(R1)
    [psi, theta, phi] = R1.as_euler('xyz')
    X0 = np.array([psi, theta, phi, estdx[0], estdx[1], estdx[2]]).astype('float64')
    # BFGS optimization:
    res = minimize(eval3Dmatchaux, X0, args=(vol1, vol2), method='BFGS', tol=1e-3,
                   options={'gtol': 1e-3, 'disp': False})
    X = res.x
    psi = X[0];
    theta = X[1];
    phi = X[2];
    Rest = R.from_euler('xyz', [psi, theta, phi], degrees=False)
    estdx = np.array([X[3], X[4], X[5]])
    return Rest


# %%
def evalO(X, R_true, R_est, G):
    psi = X[0];
    theta = X[1];
    phi = X[2];
    O = R.as_matrix(R.from_euler('xyz', [psi, theta, phi], degrees=False))
    n = np.size(G, 0)
    dist = np.zeros((1, n))
    for i in range(n):
        g = G[i, :, :]
        dist[0, i] = LA.norm(R_true - O @ g @ O.T @ R_est, 'fro')
    err = np.min(dist)
    return err


# %%
def AlignVolumes(vol1, vol2, verbose=0, opt=None):
    '''
    This function aligns vol2 according to vol1
    Aligning vol2 to vol1 by finding the relative rotation, translation and
    reflection between vol1 and vol2, such that vol2 is best aligned with
    vol1.
    How to align the two volumes:
        The user should align vol2 according to vol1 using the parameters bestR,
        bestdx and reflect. If reflect=0 then there is no reflection between the
        volumes. In that case the user should first rotate vol2 by bestR and then
        reshift by bestdx. If reflect=1, then there is a reflection between the
        volumes. In that case the user should first reflcet vol2 about the z axis
        using the flip function, then rotate the volume by bestR and finally
        reshift by bestdx.
    Input:
        vol1- 3D reference volume that vol2 should be aligned accordingly.
        vol2- 3D volume to be aligned.
        verbose- Set verbose to nonzero for verbose printouts (default is zero).
    Output:
        bestR- the estimated rotation between vol2 and vol1, such that bestR*vol2
               will align vol2 to vol1.
        bestdx- size=3x1. the estimated translation between vol2 and vol1.
        reflect- indicator for reflection. If reflect=1 then there is a
                 reflection between vol1 and vol2, else reflect=0. In order to
                 align the volumes in the case of reflect=1, the user should
                 first reflect vol2 about the z axis, and then rotate by bestR.
        vol2aligned- vol2 after applyng the estimated transformation, so it is
                     best aligned with vol1 (after optimization).
        bestcorr- the coorelation between vol1 and vol2aligned.
    Options:
        sym- the symmetry type- 'Cn'\'Dn'\'T'\'O'\'I', where n is the the
             symmetry order (for example: 'C2'). This input is required only for
             the error calculation.
        opt.downsample-  Downsample the volume to this size (in pixels) for
                         faster alignment. Default is 48. Use larger value if
                         alignment fails.
        opt.Nprojs- Number of projections to use for the alignment.
                     Defult is 30.
        opt.trueR-  True rotation matrix between vol2 and vol1, such that
                vol2 = fastrotate3d(vol1,true_R). In the case of reflection,
                true_R should be the rotation between the volumes such that
                vol2 = flip(fastrotate3d(vol1,true_R),3). In this case
                O = J*true_R, where J is the reflection matrix over the z axis
                J=diag([1,1,-1]). This input is used for debugging to calculate
                errors.
        opt.G- Array of matrices of size 3x3xn containing the symmetry group
               elemnts of vol1. This input is for accurate error calculation. If
               G is not submitted then the error will be calculated by
               optimization over the symmetry group.
    '''

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger()
    if verbose == 0: logger.disabled = True

    ### Check options:
    if hasattr(opt, 'sym'):
        sym = opt.sym
    else:
        sym = None

    if hasattr(opt, 'Nprojs'):
        Nprojs = opt.Nprojs
    else:
        Nprojs = 30

    if hasattr(opt, 'G'):
        G = opt.G
    else:
        G = None

    if hasattr(opt, 'trueR'):
        trueR = opt.trueR
    else:
        trueR = None

    if hasattr(opt, 'downsample'):
        downsample = opt.downsample
    else:
        downsample = 64

    ### Define parameters:
    sym_flag = 0
    G_flag = 0
    if sym is not None:
        s = sym[0]
        n_s = 0
        if len(sym) > 1: n_s = int(sym[1:len(sym)])
        sym_flag = 1
        if s == 'C' and n_s == 1:
            G = np.eye(3).reshape((1, 3, 3))
    if G is not None:
        G_flag = 1
    refrot = 1
    if trueR is None: refrot = 0

    # Validate input:
    # Input volumes must be 3-dimensional, where all dimensions must be equal.
    # This restriction can be remove, but then, the calculation of nr (radial
    # resolution in the Fourier domain) should be adjusted accordingly. Both
    # vol_1 and vol_2 must have the same dimensions.
    n_1 = np.shape(vol1)
    assert np.size(n_1) == 3, "Inputs must be 3D"
    assert n_1[0] == n_1[1], "All dimensions of input volumes must be equal"
    n_2 = np.shape(vol2)
    assert np.size(n_2) == 3, "Inputs must be 3D"
    assert n_2[0] == n_1[1] and n_2[0] == n_1[1], "All dimensions of input volumes must be equal"
    assert n_1[0] == n_2[0], "Input volumes have different dimensions"
    n = n_1[0]
    n_ds = min(n, downsample)  # Perform aligment on down sampled volumes.
    # This speeds up calculation, and does not seem
    # to degrade accuracy
    
    if n_ds < n:
        logger.info('Downsampling volumes from %i to %i pixels', n, n_ds)
        vol1_ds = cryo_downsample(vol1, (n_ds, n_ds, n_ds))
        vol2_ds = cryo_downsample(vol2, (n_ds, n_ds, n_ds))
    else:
        logger.info('No need for downsampling. n=%i n_ds=%i', n, n_ds)
        vol1_ds = vol1
        vol2_ds = vol2

    # Aligning the volumes:
    if G_flag == 1:
        G_c = np.copy(G)
    else:
        G_c = None
    R_est, R_est_J = fastAlignment3D(sym, vol1_ds.copy(), vol2_ds.copy(), n_ds, Nprojs, trueR, G_c, refrot, verbose);

    vol2_aligned_ds = fastrotate3d(vol2_ds, R_est)  # Rotate the original vol_2 back.
    vol2_aligned_J_ds = fastrotate3d(vol2_ds, R_est_J)

    vol2_aligned_J_ds = np.flip(vol2_aligned_J_ds, axis=2)
    estdx_ds = register_translations_3d(vol1_ds, vol2_aligned_ds)
    estdx_J_ds = register_translations_3d(vol1_ds, vol2_aligned_J_ds)
    if np.size(estdx_ds) != 3 or np.size(estdx_J_ds) != 3:
        raise Warning("***** Translation estimation failed *****")
    vol2_aligned_ds = reshift_vol(vol2_aligned_ds, estdx_ds)
    vol2_aligned_J_ds = reshift_vol(vol2_aligned_J_ds, estdx_J_ds)
    no1 = np.mean(np.corrcoef(vol1_ds.ravel(), vol2_aligned_ds.ravel(), rowvar=False)[0, 1:])
    no2 = np.mean(np.corrcoef(vol1_ds.ravel(), vol2_aligned_J_ds.ravel(), rowvar=False)[0, 1:])

    # if max(no1, no2) < 0.1:  # The coorelations of the estimated rotations are
    #     # smaller than 0.1, that is, no transformation was recovered.
    #     raise Warning("***** Alignment failed *****")
    # Do we have reflection?
    reflect = 0
    corr_v = no1
    if no2 > no1:
        J3 = np.diag([1, 1, -1])
        corr_v = no2
        R_est = R_est_J;
        R_est = J3 @ R_est @ J3;
        estdx_ds = estdx_J_ds
        vol2_ds = np.flip(vol2_ds, axis=2)
        vol2 = np.flip(vol2, axis=2)
        reflect = 1
        logger.info('***** Reflection detected *****')
    logger.info('Correlation between downsampled aligned volumes before optimization is %.4f', corr_v)
    
    
    
    if opt.no_refine:
        logger.info('Skipping refinement of alignment parameters')
        bestR = R_est
    else:
        logger.info('Using BFGS algorithm to refine alignment parameters')
        # Optimization:
        # We use the BFGS optimization algorithm in order to refine the resulted
        # transformation between the two volumes.
        bestR = refine3DmatchBFGS(vol1_ds.copy(), vol2_ds.copy(), R_est, estdx_ds)
        bestR = R.as_matrix(bestR)
        
    logger.info('Done aligning downsampled volumes')
    logger.info('Applying estimated rotation to original volumes')
    vol2aligned = fastrotate3d(vol2, bestR)
    logger.info('Estimating shift for original volumes')
    bestdx = register_translations_3d(vol1, vol2aligned)
    # if np.size(bestdx) != 3 :
    #    raise Warning("***** Translation estimation failed *****")
    
    if not opt.no_refine:
        logger.info('Refining shift for original volumes')
        bestdx = refine3DshiftBFGS(vol1, vol2, bestdx)
        
    logger.info('Translating original volumes')
    vol2aligned = reshift_vol(vol2aligned, bestdx)
    
    logger.info('Computing correlations of original volumes')
    bestcorr = np.mean(np.corrcoef(vol1.ravel(), vol2aligned.ravel(), rowvar=False)[0, 1:])

    logger.info('Estimated rotation:\n'+str(bestR))
    logger.info('Estimated translations: [%.3f, %.3f, %.3f]', bestdx[0], bestdx[1], bestdx[2])
    logger.info('Correlation between original aligned volumes is %.4f', bestcorr)
    # Accurate error calculation:
    # The difference between the estimated and reference rotation should be an
    # element from the symmetry group:
    if refrot == 1 and G_flag == 1:
        n_g = np.size(G, 0)
        g_est_t = trueR.T @ bestR.T
        dist = np.zeros((n_g))
        for g_idx in range(n_g):
            dist[g_idx] = LA.norm(g_est_t - G[g_idx, :, :], ord='fro')
        min_idx = np.argmin(dist)
        g_est = G[min_idx, :, :]
        err_norm = LA.norm(trueR.T - (g_est @ bestR), ord='fro')
        ref_true_R = trueR.T
        logger.info('Reference rotation:')
        logger.info('%.4f %.4f %.4f', ref_true_R[0, 0], ref_true_R[0, 1], ref_true_R[0, 2])
        logger.info('%.4f %.4f %.4f', ref_true_R[1, 0], ref_true_R[1, 1], ref_true_R[1, 2])
        logger.info('%.4f %.4f %.4f', ref_true_R[2, 0], ref_true_R[2, 1], ref_true_R[2, 2])
        aligned_bestR = g_est @ bestR
        logger.info('Estimated rotation (aligned by a symmetry element according to the reference rotation):')
        logger.info('%.4f %.4f %.4f', aligned_bestR[0, 0], aligned_bestR[0, 1], aligned_bestR[0, 2])
        logger.info('%.4f %.4f %.4f', aligned_bestR[1, 0], aligned_bestR[1, 1], aligned_bestR[1, 2])
        logger.info('%.4f %.4f %.4f', aligned_bestR[2, 0], aligned_bestR[2, 1], aligned_bestR[2, 2])
        logger.info('Estimated symmetry element:')
        logger.info('%.4f %.4f %.4f', g_est[0, 0], g_est[0, 1], g_est[0, 2])
        logger.info('%.4f %.4f %.4f', g_est[1, 0], g_est[1, 1], g_est[1, 2])
        logger.info('%.4f %.4f %.4f', g_est[2, 0], g_est[2, 1], g_est[2, 2])
        logger.info('Estimation error (Frobenius norm) up to symmetry group element is %.4f', err_norm)
        vec_ref = R.as_rotvec(R.from_matrix(trueR.T))
        angle_ref = LA.norm(vec_ref)
        axis_ref = vec_ref / angle_ref
        vec_est = R.as_rotvec(R.from_matrix(g_est @ bestR))
        angle_est = LA.norm(vec_est)
        axis_est = vec_est / angle_est
        logger.info('Rotation axis:')
        logger.info('Reference [ %.4f, %.4f, %.4f]', axis_ref[0], axis_ref[1], axis_ref[2])
        logger.info('Estimated [ %.4f, %.4f, %.4f]', axis_est[0], axis_est[1], axis_est[2])
        logger.info('Angle between axes is %.4f  degrees', math.degrees(np.arccos(np.dot(axis_est, axis_ref))))
        logger.info('In-plane rotation:')
        logger.info('Reference %.4f degrees', math.degrees(angle_ref))
        logger.info('Estimated %.4f degrees', math.degrees(angle_est))
        logger.info('Angle difference is %.4f degrees', abs(math.degrees(angle_ref) - math.degrees(angle_est)))
    # Error calculation by optimization:
    # The difference between the estimated and reference rotation should be an
    # element from the symmetry group. In the case the symmetry group of vol1
    # is not given, it can be estimated using optimization process.
    # Let G be the symmetry group of the symmetry type of the molecule in the
    # canonical coordinate system (G is obtained using genSymgroup). Then, the
    # symmetry group of vol1 is given by O*G*O.', where O is the orthogonal
    # transformation between the coordinate system of vol1 and the canonical
    # one. Therefore, the symmetry group can be estimated by evaluating O using
    # optimization algorithm.
    if refrot == 1 and G_flag == 0 and sym_flag == 1:
        # Creating initial guess by brute-force algorithm:
        G = genSymGroup(sym)
        n_g = np.size(G, 0)
        Rots = genRotationsGrid(75)
        n = np.size(Rots, 2)
        dist = np.zeros((n, n_g))
        for i in range(n):
            for j in range(n_g):
                O = Rots[:, :, i];
                g = G[j, :, :];
                dist[i, j] = LA.norm(trueR.T - (O @ g @ O.T @ bestR), ord='fro')
        err = np.min(dist.ravel());
        row = np.array(np.where(dist == err))[0];
        O = Rots[:, :, row[0]]
        # BFGS optimization:
        [psi, theta, phi] = (R.from_matrix(O)).as_euler('xyz')
        X0 = np.array([psi, theta, phi]).astype('float64')
        res = minimize(evalO, X0, args=(trueR.T, bestR, G), method='BFGS', tol=1e-4,
                       options={'gtol': 1e-4, 'disp': False})
        X = res.x
        psi = X[0];
        theta = X[1];
        phi = X[2];
        O = R.as_matrix(R.from_euler('xyz', [psi, theta, phi], degrees=False))
        n_g = np.size(G, 0)
        dist = np.zeros((1, n_g))
        for i in range(n_g):
            g = G[i, :, :]
            dist[0, i] = LA.norm(trueR.T - (O @ g @ O.T @ bestR), 'fro')
        err_norm = np.min(dist);
        idx = np.array(np.where(dist == err_norm));
        g = G[idx[1, 0], :, :];
        g_est = O @ g @ O.T;
        ref_true_R = trueR.T
        logger.info('Reference rotation:')
        logger.info('%.4f %.4f %.4f', ref_true_R[0, 0], ref_true_R[0, 1], ref_true_R[0, 2])
        logger.info('%.4f %.4f %.4f', ref_true_R[1, 0], ref_true_R[1, 1], ref_true_R[1, 2])
        logger.info('%.4f %.4f %.4f', ref_true_R[2, 0], ref_true_R[2, 1], ref_true_R[2, 2])
        aligned_bestR = g_est @ bestR
        logger.info('Estimated rotation (aligned by a symmetry element according to the reference rotation):')
        logger.info('%.4f %.4f %.4f', aligned_bestR[0, 0], aligned_bestR[0, 1], aligned_bestR[0, 2])
        logger.info('%.4f %.4f %.4f', aligned_bestR[1, 0], aligned_bestR[1, 1], aligned_bestR[1, 2])
        logger.info('%.4f %.4f %.4f', aligned_bestR[2, 0], aligned_bestR[2, 1], aligned_bestR[2, 2])
        logger.info('Estimated symmetry element:')
        logger.info('%.4f %.4f %.4f', g_est[0, 0], g_est[0, 1], g_est[0, 2])
        logger.info('%.4f %.4f %.4f', g_est[1, 0], g_est[1, 1], g_est[1, 2])
        logger.info('%.4f %.4f %.4f', g_est[2, 0], g_est[2, 1], g_est[2, 2])
        logger.info('Estimation error (Frobenius norm) up to symmetry group element is %.4f', err_norm)
        vec_ref = R.as_rotvec(R.from_matrix(trueR.T))
        angle_ref = LA.norm(vec_ref)
        axis_ref = vec_ref / angle_ref
        vec_est = R.as_rotvec(R.from_matrix(g_est @ bestR))
        angle_est = LA.norm(vec_est)
        axis_est = vec_est / angle_est
        logger.info('Rotation axis:')
        logger.info('Reference [ %.4f, %.4f, %.4f]', axis_ref[0], axis_ref[1], axis_ref[2])
        logger.info('Estimated [ %.4f, %.4f, %.4f]', axis_est[0], axis_est[1], axis_est[2])
        logger.info('Angle between axes is %.4f  degrees', math.degrees(np.arccos(np.dot(axis_est, axis_ref))))
        logger.info('In-plane rotation:')
        logger.info('Reference %.4f degrees', math.degrees(angle_ref))
        logger.info('Estimated %.4f degrees', math.degrees(angle_est))
        logger.info('Angle difference is %.4f degrees', abs(math.degrees(angle_ref) - math.degrees(angle_est)))

    logging.shutdown()

    return bestR, bestdx, reflect, vol2aligned, bestcorr
