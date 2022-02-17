import numpy as np
from scipy.io import loadmat, savemat

def mat_to_npy(file_name):
    if '.mat' not in file_name:
        file_name += '.mat'
    full_mat = loadmat(file_name )
    key = None
    for k in full_mat:
        if '__' not in k:
            key = k
    return full_mat[key]


def npy_to_mat(file_name, var_name, var):
    if '.mat' in file_name:
        file_name = file_name[:-4]
    savemat(file_name, {var_name: var})


def mat_to_npy_vec(file_name):
    a = mat_to_npy(file_name)
    return a.reshape(a.shape[0] * a.shape[1])


def comp(a, b):
    norm_diff = np.linalg.norm(a - b)
    norm_orig = np.linalg.norm(a)
    return norm_diff/norm_orig
