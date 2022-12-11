#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 22:04:50 2022

@author: yoel
"""

import numpy as np
import logging
import os
import uuid
import subprocess
import scipy.spatial.transform
import math
import time
import shlex
import parse
import pandas as pd
import matplotlib.pyplot as plt
import shutil

import src.cryo_fetch_emdID
import src.rand_rots
import src.read_write
import src.common_finufft
import src.fastrotate3d
import src.reshift_vol
import src.SymmetryGroups
import src.fsc

test_densities = [
    ['C1',     '2660',    1.34],
    ['C2',     '0667',    1.38],
    ['C3',     '0731',    0.88],
    ['C4',     '0882',    1.45],
    ['C5',    '21376',    0.91],
    ['C7',    '11516',    0.646],
    ['C8',    '21143',    1.06],
    ['C11',    '6458',    0.86],
    ['D2',    '30913',    0.7999967],
    ['D3',    '20016',    0.83],
    ['D4',    '22462',    0.844],
    ['D7',     '9233',    0.66159993],
    ['D11',   '21140',    1.06],
    ['T',      '4179',    0.97],
    #['O',     '22658',    0.502],
    ['I',     '24494',    1.06 ]
]


#emalign_cmd = "/home/yoelsh/.local/bin/emalign"
emalign_cmd = shutil.which('emalign')

# Setup logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()

# Save both to file and console
fh = logging.FileHandler('./emalign.log')
fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.addHandler(fh)


def init_random_state():
    '''
    Initialize random number generators for reproducible results
    '''
    random_state = ('MT19937',
       [1221878109,   75677465, 1084386505, 3766507770, 3249015377,
        2484651255, 2842570038, 3761918232, 4019838424,  914489418,
        1780152765,  446073632, 3149352333,  432466234,  612102031,
         807601325,  659379188,    5708672, 2085154143, 4203578507,
        1170444074, 3016922801, 3694734902, 1550909049,  352895502,
        1166868841,  940610656, 1558543108, 1349073032, 1055620743,
        3452683944, 2498401103, 1477012983, 2576869499, 3739830677,
        2940976410, 2791066130, 1743530211,  639966781,   78850694,
        3029342349,  221561436, 3636361266, 1900941679, 4079807434,
         858058223,  930483010, 1056360068, 2519149397, 3361732692,
        3383730835,  700940861,  136479548, 4042953938, 3233474165,
        1774837099, 1358192185,  450016557, 2110812368,  533776066,
        3447734058,  648674185, 4003519104, 2750399302, 2922582061,
        2541595169, 1319257470,  474767093, 4057682953, 2351300709,
        3421337997, 2943459859,  855745003, 3186082196, 1159986367,
        4108292614, 4266003132, 3255401266,  528806712, 4053835497,
        3676428708, 3813703787, 4045140135, 4063077667, 1700119915,
        2607155689, 2541463547, 2512422466, 2554451783,  964625116,
        4002621892, 1755423192, 2080162550, 2775799247, 4151848463,
        1427060721, 2207893147, 3140946204, 4142355920, 3614163794,
        4010260661, 2071932252, 3292276704, 2491863311, 3136458899,
        2502220832, 3217394738,  102887599, 3034113730, 3685201660,
        3831779428, 2894444711, 1505053255, 4025709891, 2859466680,
        3647371779,  284801883, 2318903588, 4063679370, 2180046373,
        2453326967,  511477416, 2542954617, 3313287685, 2179175177,
        3414263133, 1948217064, 2550258943, 3865779938, 2587451685,
        1945904160, 2027985202, 1859867893, 3808075129, 3715408424,
        4005178980,  246657870, 2933734354,  326618583, 3637268606,
        2221986077, 4264014091, 2994100509, 1998543850, 3496324599,
        4035610278, 2639127501,  453780866, 3780348870, 1525365924,
        3425226408, 1816810312, 1184740620, 2292765088,  368838827,
        4294049803,   62261911, 2451727714,  252229268, 3920055310,
        4079776978, 1040042561, 1184635004, 2854663540, 3414456266,
        1456157567, 2329433583,   63322392, 1786919459, 3471589524,
        2547267784, 3857042923, 3687614333,  896538967, 2065453559,
        1484732893, 2468713596, 3829258790, 1913690668, 2782358542,
        3750115123, 3831068342, 3254979715,  649077774, 1638545612,
        3204405079, 1584485197,  718861853,  891221044, 3435849578,
        4118879645,  734767970,  386857851, 1234937415, 2262843648,
        1890750987, 2260469486, 3376298237,  558433045,  384682340,
        3282739904, 3865377344, 1526231695, 1132299599, 1072355399,
        3551375914,    4250304, 4125550148, 2634730291, 4188332788,
        4087782438, 1149773393,  433855413, 3843450137, 2198943377,
        1937912555, 2956065675, 2735755748, 3209968191, 3775382441,
        1543496565, 3286448607, 4246134666, 3453388739, 3867069436,
        2772545793, 1717000800, 2544622482,  912665192, 1654203654,
        3479988059, 3867396577, 1946762708, 1252315168, 2165917678,
        2334069649, 1688178898, 3804150450, 3050027511, 2076670514,
        2895769660, 2885621455, 3970141300, 1248446439, 4191930926,
        2731286115,  543562289, 3920538768,  107958942, 2645836043,
        3813757265, 3844566409, 3796550428,  780959547,  362491015,
        1549131671, 3702025373, 3524230831, 1633182084, 1735214783,
        1201354805, 1331119464, 1529356360, 1737445510,  593888807,
        2611567910, 3616921353, 3721402761, 3741493971, 3422578320,
        2077726336, 2863435954, 3971255462, 2628528855, 1276309573,
        1438409957,  847511981, 2789904466, 3148737069, 1185579156,
        1480264641,  540994998, 3628616995,  926746308, 1120985889,
        1014560863,  700483183,  309003198, 3983269458, 2716771974,
        3186967474, 1614111024, 2756801176, 3146244691, 3597410383,
        3910584991, 1782676571,  382311861,  311918108, 2475358803,
        2438197710,  594593872, 1320745707,  685314732, 2781204379,
         934983486,  565914795, 1763747950, 2980760079, 1912104892,
        1929596252, 2169515538, 2126013902,  685956549, 1687010797,
        3398376886, 3534848416,  629772814, 1939635443, 1507393584,
        1761125494, 3320986349, 2026425240, 2143079557, 3219072010,
        3570390855, 3665276483,  775335683, 3384876795, 3267316067,
        3014325430, 2369829029, 1957057776, 3249775512, 4137994080,
        3419335788, 3097565343,  877933787, 3090782526, 3897760458,
        2384574564, 2906924700,  133334534,  902820749, 1952151834,
        1912499886,  143772262,  764203746,    3640429, 2614345439,
        3278582340, 1637351462,  656838755, 2112962062, 4273338958,
        4112566094, 3454697352, 4095816286, 2828178138, 2229342696,
         851460209, 3928016412, 2230986225,  505720189, 3104986739,
        2079560367, 3468516434, 2100660503, 2059622749,  582792595,
        1507779749, 1923912117, 1953798289, 1513745002, 2351792370,
        3130414952, 3738441186, 3355275942, 2083368177, 2294510970,
        3708844669, 2790400894, 2164194065, 1075421339, 3564458171,
        1879004704, 4089229499, 3791226688, 3668163254, 2445481440,
        3839573359, 2440166596, 2083395284, 3974937840,   75332406,
        2113709682,  896022953, 3136667294, 2424047364, 3902638420,
        2057570957, 1281770901, 1004667493,   59129503, 3407276051,
         851581873, 2115377077, 1839773569, 3757487132, 2571857115,
         876095603, 4180960305, 4050851797, 1489238616, 2568160855,
        3905813784, 3267955020, 3274665467,   60201274, 1483204190,
        3644523429, 2648418916, 2054224260, 1132479285, 2029753738,
         443070591, 1936096977, 2228842556,   29626418, 3865232917,
        3061734957,   56422697, 2210865701,  612763221, 2430455834,
        1087597130, 1259496928,  684613686,  795496926, 1698064543,
        3606612530,   18957446,  586442408, 2396435000,  608794084,
        3286906349,  856159875,  241319307, 2595444609, 1908205087,
        1190418072, 3758723303, 3168603784, 3968108531, 2477995114,
        1470088006, 2495933193, 3102333821, 1398996840, 2834669352,
        3299293361, 3301342853, 2270011739, 1393883381, 3705071892,
        2882638429, 4160196953,  256741283, 2141448113, 2598383216,
        4069659963, 1892248252, 3141402135, 3876460150, 3640453325,
        4034482595, 3667239663, 2498226120, 1825715048, 1351019243,
         456299072, 3920521253, 2759149308, 1792229261,  146632916,
        3886350962, 3778592573,  281358423, 3274200430, 2626262525,
        3957402528, 2467721596, 1431448484,  824947828, 2661050646,
        3230798340, 1093853904,  277156521,  428381958, 1387575789,
        3710419110, 1837551645, 1892938906, 1346444244, 1600238848,
        2990813495, 2913139748,  481813274,  996136825, 3986213884,
          22841581, 2701878314, 1638758159,  279116909,  473394613,
        1982970915, 2501960150, 2582632447, 3494167478, 2318196402,
        3030434202, 3860231229, 2429982368,  510458273, 2031998415,
        1586954600,   75060399, 2886998222, 3539848260, 1683686803,
        2022012759,  246189369, 3168037490,  532581673, 2636647425,
        3238745337, 3760649392, 1471589050, 2779786813, 2549859917,
         169321630, 2761987091, 1469518458,  114874482, 3735629244,
        1784621104, 3610287058, 2088112559,  319329845, 3219227386,
        2609397286,  640833036,  515398690, 4196195648, 2815720962,
        1764644159, 2514888212, 3166468127, 2975136546, 1358178958,
        2320544821, 2358398829, 3099537265, 1333653731,  528861567,
        3649801600, 3146980913,  386251947, 2871088599, 2443996988,
        4151573705, 1472447445,   96356160, 3614951411, 1279822533,
        3765131489, 3031618896,  338446964,  881097773, 4096810421,
        3280455258, 3506272785, 3292066781,  172765125, 2565352953,
        1544042676, 1828724193,   94761667,  172631780,   18874824,
        1607761557, 1004961020, 1209715336, 3763586663,  240420497,
         490298683, 2178598460, 2277462740,    7270124,  163072778,
        2155857045, 2446164586, 3938304807, 1022295064, 1505121556,
         624482720, 1782152468,  757174172, 1502614853, 1337164619,
        2600168882, 1256372140,  757182475, 3442989097, 2686223822,
         797512348, 3422036853,  518235075, 3010496474, 4151109974,
        1338511824, 1748122349, 2964871821, 2282404312],
     403,
     0,
     0.0)

    np.random.set_state(random_state)


def get_test_filenames(working_dir, emdid):
    '''
    Returns the names of the filenames for a given emdid.
    The returned value is a dictionary with the following entries
       ref: 'map_EMDID_ref.mrc'
       transformed: 'map_EMDID_transformed.mrc',
       aligned_norefine: 'map_EMDID_aligned_norefine.mrc',
       aligned_refine: 'map_EMDID_aligned_refine.mrc'       
    '''
    
    dict={}
    dict['ref'] = os.path.join(working_dir,'map_{0:s}_ref.mrc'.format(emdid))
    dict['transformed'] = os.path.join(working_dir,'map_{0:s}_transformed.mrc'.format(emdid))
    dict['aligned_norefine'] = os.path.join(working_dir,'map_{0:s}_aligned_norefine.mrc'.format(emdid))
    dict['aligned_refine'] = os.path.join(working_dir,'map_{0:s}_aligned_refine.mrc'.format(emdid))    
    
    return dict


def download_data(working_dir):
    '''
    Download density maps specified in test_densities from EMDB.
    The densities are saved in working_dir.
    If working_dir is not empty, the function aborts.
    The functions creates working_dir if it does not exist.

    Parameters
    ----------
    working_dir : string
        Directory to save the downlaoded density maps.
        
    Returns
    -------
    None.

    '''

    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
    else: 
        dir = os.listdir(working_dir)
  
        # Checking if the list is empty or not
        if len(dir) != 0:
            raise ValueError('Working directory is not empty.')

    
        for testidx in range(len(test_densities)):
    
            # Download density map    
            test_data = test_densities[testidx]
            symmetry = test_data[0]
            emdid = test_data[1]

            logger.info('Downloading %d/%d %s  (EMD%s)',testidx+1,len(test_densities),symmetry,emdid)
            
            fnames_dict = get_test_filenames(working_dir,emdid)
            map_name = fnames_dict['ref']            
            src.cryo_fetch_emdID.cryo_fetch_emdID(emdid,map_name)


def rot_from_params_file(fname):
    ''' Read rotation from paramters file'''    
    Rest = np.zeros([3,3])
    
    try:
        file = open(fname)
    
        lines = file.readlines() [-3:]
        i = 0
        for line in (lines):
        # Remove [ and ] from parameters file
            line = (line.translate({ord('['):None, ord(']'):None}))
            line = line.strip()
            line = line.split()
                                       
            for j in range(3):
                Rest[i][j] = float(line[j])

            i = i +1
    finally:
        file.close()
        
    return Rest

def measure_error(R,Rest,symmetry):
    '''
    Calculate the difference between the estimated and ground truth rotation 
    up to an elemenet of the symmetry group.

    Parameters
    ----------
    R : ndarray
        3x3 numpy nd array representing the ground truth rotation.
    Rest : ndarray
        3x3 numpy nd array representing the estimated rotation.
    symmetry : string
        Symmetry group

    Returns
    -------
    err_ang1 : float
        Angle in degrees between the rotation axes of the ground truth and 
        estimated rotations.
    err_ang2 : float
        Angle in degrees between the rotation angle around the axis of the 
        ground truth and estimated rotations.

    '''
            
    G = src.SymmetryGroups.genSymGroup(symmetry)
    n_g = G.shape[0]
    g_est_t = R.transpose() @ Rest
    dist = np.zeros(n_g)
    for g_idx in range(n_g):
        dist[g_idx] = np.linalg.norm(G[g_idx]-g_est_t,'fro')

    min_idx = np.argmin(dist)
    g_est = G[min_idx]
            
    tmp_R = scipy.spatial.transform.Rotation.from_matrix(R.transpose())
    tmp_rotvec = tmp_R.as_rotvec()            
    axis_ref = tmp_rotvec/np.linalg.norm(tmp_rotvec)
    angle_ref = np.linalg.norm(tmp_rotvec)

    tmp_R = scipy.spatial.transform.Rotation.from_matrix(g_est @ Rest.transpose())
    tmp_rotvec = tmp_R.as_rotvec()            
    axis_est = tmp_rotvec/np.linalg.norm(tmp_rotvec)
    angle_est = np.linalg.norm(tmp_rotvec)
    
    err_ang1 = math.acos(np.dot(axis_ref,axis_est))/math.pi*180
    
    if err_ang1 > 90:
        axis_est = -axis_est
        angle_est = 2*np.pi - angle_est
        err_ang1 = math.acos(np.dot(axis_ref,axis_est))/math.pi*180
    
    err_ang2 = abs((np.arcsin(abs(np.exp(complex(0,1)*angle_ref)
                -np.exp(complex(0,1)*angle_est))/2)*2)*180/np.pi)

    return err_ang1, err_ang2

def run_cmd(cmd):
    '''
    Run shell command

    Parameters
    ----------
    cmd : string
        Command to run.

    Returns
    -------
    t : float
        Execution time.

    '''
    
    t_start = time.time()
    
    # Run without printouts
    with open('emalign.log', "a") as outfile:
        subprocess.run(shlex.split(cmd), stdout=outfile, stderr=outfile)  
    
    # Run with printouts
    # p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, 
    #                      shell=False, bufsize = 0, close_fds = True)
    # # Grab stdout line by line as it becomes available.  
    # # This will loop until p terminates.
    # # while p.poll() is None:
    # #     l = p.stdout.readline() # This blocks until it receives a newline.
    # #     print(l)
    # #     # When the subprocess terminates there might be unconsumed output 
    # #     # that still needs to be processed.
    # #     print(p.stdout.read())
        
    # while True:        
    #     output = p.stdout.readline()
    #     if output == '' and p.poll() is not None:
    #         break
    #     if output:
    #         print(output.strip())
    # print(p.stdout.read())
    # p.stdout.close()
    # p.wait()
            
    t_end = time.time()
    return t_end-t_start

#%%
def results_varying_downsampling():
    init_random_state()
    
    working_dir = './varying_downsampling'
    data_dir = './data/'    
    results = []    
    sizes = [16, 32, 64, 128]  # Sizes of downsampled volumes
    #sizes = [16, 32, 64]  # Sizes of downsampled volumes
        
    for sz_ds in sizes:
        
        #sz_ds = 64 # Size of downsampled volume
    
        for testidx in range(len(test_densities)):
        
            #Generate two density maps
        
            test_data = test_densities[testidx]
            symmetry = test_data[0]
            emdid = test_data[1]        
            fnames_dict = get_test_filenames(data_dir, emdid)
    
            logger.info('Test %d/%d %s  (EMD%s)',testidx+1,
                        len(test_densities),symmetry,emdid)
            
            vol = src.read_write.read_mrc(fnames_dict['ref'])            
            #vol = src.common_finufft.cryo_downsample(vol,[64,64,64])
            
            
            sz_orig = vol.shape[0]  # Size of original volume (before downsampling)

            # Generate a random rotation
            R = np.squeeze(src.rand_rots.rand_rots(1))
            assert abs(np.linalg.det(R)-1) <1.0e-8
            logger.info("R_ref = \n"+str(R))

            # Rotate the reference volume by the random rotation
            vol_transformed = src.fastrotate3d.fastrotate3d(vol,R)
            
            # Add shift of up to 10%
            shift = np.floor((np.random.rand(3)-1/2)*sz_orig*0.1)
            vol_transformed = src.reshift_vol.reshift_vol(vol_transformed,shift)
            
            try:                
                # Save the two volues to align
                vol_ref_name = str(uuid.uuid4())+'.mrc'
                vol_ref_name = os.path.join(working_dir, vol_ref_name)
                src.read_write.write_mrc(vol_ref_name, vol)
    
                vol_rot_name = str(uuid.uuid4())+'.mrc'
                vol_rot_name = os.path.join(working_dir, vol_rot_name)
                src.read_write.write_mrc(vol_rot_name, vol_transformed)
                
                vol_aligned_norefine_name = str(uuid.uuid4())+'.mrc'
                vol_aligned_norefine_name = os.path.join(working_dir,\
                                                         vol_aligned_norefine_name)
    
                vol_aligned_refine_name = str(uuid.uuid4())+'.mrc'
                vol_aligned_refine_name = os.path.join(working_dir,\
                                                         vol_aligned_refine_name)
    
                params_norefine_name = str(uuid.uuid4())+'.txt'
                params_refine_name = str(uuid.uuid4())+'.txt'
                
                
                ####
                # Run alignment without refiment
                ####
                align_cmd = emalign_cmd + (' --vol1 {0:s} --vol2 {1:s} '+
                '--output-vol {2:s} --downsample {3:d} '+
                '--n-projs {4:d} '+
                '--output-parameters {5:s} --no-refine '+
                '--verbose').format(vol_ref_name, vol_rot_name,
                                    vol_aligned_norefine_name, sz_ds, 30,
                                    params_norefine_name)                         
           
                                    
                # Run alignment command
                t_norefine = run_cmd(align_cmd)
    
                # Read estimated matrix from paramters file
                Rest = rot_from_params_file(params_norefine_name)
                Rest = Rest.transpose()
                
                # Calculate error between ground-truth and estimated rotation
                err_ang1_norefine, err_ang2_norefine = measure_error(R, Rest, 
                                                                     symmetry)
    
    
                ####
                # Run alignment with refiment
                ####
                align_cmd = emalign_cmd + (' --vol1 {0:s} --vol2 {1:s} '+
                '--output-vol {2:s} --downsample {3:d} '+
                '--n-projs {4:d} '+
                '--output-parameters {5:s}  '+
                '--verbose').format(vol_ref_name, vol_rot_name, 
                                    vol_aligned_refine_name, sz_ds, 30,
                                    params_refine_name)                                
                
                # Run alignment command
                t_refine = run_cmd(align_cmd)
    
                # Read estimated matrix from paramters file
                Rest = rot_from_params_file(params_refine_name)
                Rest = Rest.transpose()
                
                # Calculate error between ground-truth and estimated rotation
                err_ang1_refine, err_ang2_refine = measure_error(R, Rest, 
                                                                 symmetry)
    
                    
                test_result = [symmetry, emdid, sz_orig, sz_ds,
                               err_ang1_norefine, err_ang2_norefine, 
                               t_norefine, err_ang1_refine, err_ang2_refine, 
                               t_refine]
                                    
                logger.info(test_result)
                results.append(test_result)
                # sz_ds = sz_ds_sav  # Restore current downsampling
                
            # Cleanup
            finally:
                if os.path.exists(vol_ref_name):
                    os.remove(vol_ref_name)
                    
                if os.path.exists(vol_rot_name):
                    os.remove(vol_rot_name)
                            
                if os.path.exists(vol_aligned_norefine_name):
                    os.remove(vol_aligned_norefine_name)
    
                if os.path.exists(vol_aligned_refine_name):
                    os.remove(vol_aligned_refine_name)
    
                if os.path.exists(params_norefine_name):
                    os.remove(params_norefine_name)
    
                if os.path.exists(params_refine_name):
                    os.remove(params_refine_name)
    
    df = pd.DataFrame(results, columns = ['symmetry','emdid','size_orig', 
                'size_ds','err_ang1_norefine','err_ang2_norefine',
                't_norefine','err_ang1_refine','err_ang2_refine','t_refine'])
    df.to_csv("results_varying_downsampling.txt")
    df.to_excel("results_varying_downsampling.xlsx")


    #return results



#%%
def results_varying_Nprojs():
    init_random_state()
    
    
    working_dir = './varying_nprojs'
    data_dir = './data'
    results = []    
    # n_projs_list = [3, 10, 15, 20, 25, 30, 100]  # Number of reference projections
    n_projs_list = [10, 30, 50, 70, 90]  
        
    for n_projs in n_projs_list:
        
        sz_ds = 64 # Size of downsampled volume
    
        for testidx in range(len(test_densities)):
        #for testidx in range(1):
            #Generate two density maps
        
            test_data = test_densities[testidx]
            symmetry = test_data[0]
            emdid = test_data[1]        
            fnames_dict = get_test_filenames(data_dir, emdid)
            logger.info('Test %d/%d %s  (EMD%s)',testidx+1,
                        len(test_densities),symmetry,emdid)
            
            vol = src.read_write.read_mrc(fnames_dict["ref"])            
            #vol = src.common_finufft.cryo_downsample(vol,[64,64,64])
            
            
            sz_orig = vol.shape[0]  # Size of original volume (before downsampling)

            # Generate a random rotation
            R = np.squeeze(src.rand_rots.rand_rots(1))
            assert abs(np.linalg.det(R)-1) <1.0e-8


            # Rotate the reference volume by the random rotation
            vol_transformed = src.fastrotate3d.fastrotate3d(vol,R)
            
            # Add shift of up to 10%
            shift = np.floor((np.random.rand(3)-1/2)*sz_orig*0.1)
            vol_transformed = src.reshift_vol.reshift_vol(vol_transformed,shift)
            
            try:                
                # Save the two volues to align
                vol_ref_name = str(uuid.uuid4())+'.mrc'
                vol_ref_name = os.path.join(working_dir, vol_ref_name)
                src.read_write.write_mrc(vol_ref_name, vol)
    
                vol_rot_name = str(uuid.uuid4())+'.mrc'
                vol_rot_name = os.path.join(working_dir, vol_rot_name)
                src.read_write.write_mrc(vol_rot_name, vol_transformed)
                
                vol_aligned_norefine_name = str(uuid.uuid4())+'.mrc'
                vol_aligned_norefine_name = os.path.join(working_dir,\
                                                         vol_aligned_norefine_name)
    
                vol_aligned_refine_name = str(uuid.uuid4())+'.mrc'
                vol_aligned_refine_name = os.path.join(working_dir,\
                                                         vol_aligned_refine_name)
    
                params_norefine_name = str(uuid.uuid4())+'.txt'
                params_refine_name = str(uuid.uuid4())+'.txt'
                
                
                ####
                # Run alignment without refiment
                ####
                align_cmd = emalign_cmd + (' --vol1 {0:s} --vol2 {1:s} '+
                '--output-vol {2:s} --downsample {3:d} '+
                ' --n-projs {4:d} ' + 
                '--output-parameters {5:s} --no-refine '+
                '--verbose').format(vol_ref_name, vol_rot_name,
                                    vol_aligned_norefine_name, sz_ds, n_projs,
                                    params_norefine_name)                         
           
                                    
                # Run alignment command
                t_norefine = run_cmd(align_cmd)
    
                # Read estimated matrix from paramters file
                Rest = rot_from_params_file(params_norefine_name)
                Rest = Rest.transpose()
                
                # Calculate error between ground-truth and estimated rotation
                err_ang1_norefine, err_ang2_norefine = measure_error(R, Rest, 
                                                                     symmetry)
    
    
                ####
                # Run alignment with refiment
                ####
                align_cmd = emalign_cmd + (' --vol1 {0:s} --vol2 {1:s} '+
                '--output-vol {2:s} --downsample {3:d} '+
                ' --n-projs {4:d} ' + 
                '--output-parameters {5:s}  '+
                '--verbose').format(vol_ref_name, vol_rot_name, 
                                    vol_aligned_refine_name, sz_ds, n_projs,
                                    params_refine_name)                                
                
                # Run alignment command
                t_refine = run_cmd(align_cmd)
    
                # Read estimated matrix from paramters file
                Rest = rot_from_params_file(params_refine_name)
                Rest = Rest.transpose()
                
                # Calculate error between ground-truth and estimated rotation
                err_ang1_refine, err_ang2_refine = measure_error(R, Rest, 
                                                                 symmetry)
    
                    
                test_result = [symmetry, emdid, sz_orig, n_projs,
                               err_ang1_norefine, err_ang2_norefine, 
                               t_norefine, err_ang1_refine, err_ang2_refine, 
                               t_refine]
                                    
                print(test_result)
                results.append(test_result)
                # sz_ds = sz_ds_sav  # Restore current downsampling
                
            # Cleanup
            finally:
                if os.path.exists(vol_ref_name):
                    os.remove(vol_ref_name)
                    
                if os.path.exists(vol_rot_name):
                    os.remove(vol_rot_name)
                            
                if os.path.exists(vol_aligned_norefine_name):
                    os.remove(vol_aligned_norefine_name)
    
                if os.path.exists(vol_aligned_refine_name):
                    os.remove(vol_aligned_refine_name)
    
                if os.path.exists(params_norefine_name):
                    os.remove(params_norefine_name)
    
                if os.path.exists(params_refine_name):
                    os.remove(params_refine_name)

    df = pd.DataFrame(results, columns = ['symmetry','emdid','size_orig', 'n_projs','err_ang1_norefine','err_ang2_norefine','t_norefine','err_ang1_refine','err_ang2_refine','t_refine'])
    df.to_csv("results_varying_nprojs.txt")
    df.to_excel("results_varying_nprojs.xlsx")

    return results


#%%
def results_comparison_to_eman():
    init_random_state()
    
    disable_preprocess = True
    disable_analysis = False
    
    if not disable_preprocess:
        # Create EMAN script file
        eman_script =  open("run_eman_test.sh", 'w')
        emalign_script =  open("run_emalign_test.sh", 'w')

    results = []
    
    for testidx in range(len(test_densities)):
        
        #Generate two density maps
        
        test_data = test_densities[testidx]
        symmetry = test_data[0]
        emdid = test_data[1]        
        
        # Generate filenames
        eman_dir = "./comparison_to_eman"            
        fnames_dict={}
        fnames_dict['ref'] = os.path.join("data",'map_{0:s}_ref.mrc'.format(emdid))
        fnames_dict['ref_copy'] = os.path.join(eman_dir,'map_{0:s}_ref.mrc'.format(emdid))
        fnames_dict['transformed'] = os.path.join(eman_dir,'map_{0:s}_transformed.mrc'.format(emdid))
        fnames_dict['aligned_norefine'] = os.path.join(eman_dir,'map_{0:s}_aligned_norefine.mrc'.format(emdid))
        fnames_dict['aligned_refine'] = os.path.join(eman_dir,'map_{0:s}_aligned_refine.mrc'.format(emdid))
        fnames_dict['aligned_eman'] = os.path.join(eman_dir,'map_{0:s}_aligned_eman.mrc'.format(emdid))
        fnames_dict['ref_rot'] = os.path.join(eman_dir,'ref_rot_{0:s}.txt'.format(emdid))
        fnames_dict['output_eman'] = os.path.join(eman_dir,'params_eman_{0:s}.txt'.format(emdid))        
        fnames_dict['timing_eman'] = os.path.join(eman_dir,'timing_eman_{0:s}.txt'.format(emdid))
        fnames_dict['output_norefine'] = os.path.join(eman_dir,'output_norefine_{0:s}.txt'.format(emdid))
        fnames_dict['output_refine'] = os.path.join(eman_dir,'output_refine_{0:s}.txt'.format(emdid))
        fnames_dict['timing_norefine'] = os.path.join(eman_dir,'timing_norefine_{0:s}.txt'.format(emdid))
        fnames_dict['timing_refine'] = os.path.join(eman_dir,'timing_refine_{0:s}.txt'.format(emdid))
                
        logger.info('Test %d/%d %s  (EMD%s)',testidx+1,
                            len(test_densities),symmetry,emdid)
            
        if not disable_preprocess:
            vol = src.read_write.read_mrc(fnames_dict['ref'])            
                            
            sz_orig = vol.shape[0]  # Size of original volume (before downsampling)
    
            # Generate a random rotation
            R = np.squeeze(src.rand_rots.rand_rots(1))
            assert abs(np.linalg.det(R)-1) <1.0e-8
           
            # Rotate the reference volume by the random rotation
            vol_transformed = src.fastrotate3d.fastrotate3d(vol,R)
                
            # Add shift of up to 10%
            shift = np.floor((np.random.rand(3)-1/2)*sz_orig*0.1)
            vol_transformed = src.reshift_vol.reshift_vol(vol_transformed,shift)
            
            # Save volumes to align        
            src.read_write.write_mrc(fnames_dict["ref_copy"], vol)                
            src.read_write.write_mrc(fnames_dict["transformed"], vol_transformed)                
    
            # Save rotation paramters    
            with open(fnames_dict["ref_rot"], 'w') as f:
                f.write(str(R)+"\n")
                    
                # Generate EMAN script        
       
            eman_script.write("start=`date +%s`\n")
            eman_script.write(("e2proc3d.py {0:s} {1:s} --alignref {2:s} " + 
                          "--align rotate_translate_3d_tree "+
                          "--verbose 1 > {3:s}\n"
                          ).format(fnames_dict["transformed"],
                                   fnames_dict["aligned_eman"],
                                   fnames_dict["ref"], fnames_dict["output_eman"]))
            eman_script.write("end=`date +%s`\n")
            eman_script.write(("echo `expr $end - $start` > {0:s}\n\n"
                               ).format(fnames_dict["timing_eman"]))
        
            eman_script.flush()
            
            ###################################################################
            # You have to run the EMAN script run_eman.sh from 
            # command line, It is not possible ti run it from within Pyhton 
            # since it uses a different conda environment
            ###################################################################
        
        
            # Generate emalign script
            sz_ds = 64
            n_projs = 30
                    
            # Run alignment without refiment                    
            align_cmd = emalign_cmd + (' --vol1 {0:s} --vol2 {1:s} '+
                '--output-vol {2:s} --downsample {3:d} '+
                '--n-projs {4:d} '+
                '--output-parameters {5:s} --no-refine '+
                '--verbose').format(fnames_dict['ref_copy'], 
                                    fnames_dict['transformed'],
                                    fnames_dict['aligned_norefine'], sz_ds, 
                                    n_projs, fnames_dict['output_norefine'])
                                        
            emalign_script.write("start=`date +%s`\n")
            emalign_script.write(align_cmd+"\n")
            emalign_script.write("end=`date +%s`\n")
            emalign_script.write(("echo `expr $end - $start` > {0:s}\n"
                                   ).format(fnames_dict["timing_norefine"]))
            emalign_script.write("\n")
       
        
            # Run alignment with refiment
            align_cmd = emalign_cmd + (' --vol1 {0:s} --vol2 {1:s} '+
                '--output-vol {2:s} --downsample {3:d} '+
                '--n-projs {4:d} '+
                '--output-parameters {5:s} '+
                '--verbose').format(fnames_dict['ref_copy'], fnames_dict['transformed'],
                                    fnames_dict['aligned_refine'], sz_ds, 
                                    n_projs, fnames_dict['output_refine'])
                                    
            emalign_script.write("start=`date +%s`\n")
            emalign_script.write(align_cmd+"\n")
            emalign_script.write("end=`date +%s`\n")
            emalign_script.write(("echo `expr $end - $start` > {0:s}\n"
                                  ).format(fnames_dict["timing_refine"]))
            emalign_script.write("\n\n")
        
            emalign_script.flush()
            
        if not disable_analysis:
        # Parse outputs
        
            R_ref = rot_from_params_file(fnames_dict["ref_rot"])        
            
            # Eman accuracy
            with open(fnames_dict["output_eman"], 'r') as f:
                lines = f.readlines() 
                res = (parse.search("'az':{0:g},'alt':{},'phi':{},'tx'",lines[1]))
                az = float(res[0])
                alt = float(res[1])
                phi = float(res[2])
                
                # It seems that 
                # (scipy.spatial.transform.Rotation.from_matrix(R.transpose())).as_euler('zyz',degrees=True)
                # is equivalent to [az,alt,phi-180].
                # In other words, 
                # (scipy.spatial.transform.Rotation.from_euler('zyz',[float(az),float(alt),float(phi)-180], degrees=True)).as_matrix()-R.transpose()
                # is small
                
                R_eman = ((scipy.spatial.transform.Rotation.from_euler('zyz',[float(az),float(alt),float(phi)], degrees=True)).as_matrix()).transpose()
                err_ang1_eman, err_ang2_eman = measure_error(R_ref,R_eman,symmetry)
                
            # Eman Timing
            with open(fnames_dict["timing_eman"], 'r') as f:
                t_eman = float(f.readline())
              
            # Emalign accuracy
            R_norefine = rot_from_params_file(fnames_dict["output_norefine"])
            err_ang1_norefine, err_ang2_norefine = measure_error(R_ref, 
                                            R_norefine.transpose(), symmetry)


            R_refine = rot_from_params_file(fnames_dict["output_refine"])
            err_ang1_refine, err_ang2_refine = measure_error(R_ref, 
                                R_refine.transpose(), symmetry)


            # Emalign timing
            with open(fnames_dict["timing_norefine"], 'r') as f:
                t_norefine = float(f.readline())
              
            with open(fnames_dict["timing_refine"], 'r') as f:
                t_refine = float(f.readline())
    
            
            # Save results
            vol_ref = src.read_write.read_mrc(fnames_dict["ref"])
            sz_orig = vol_ref.shape[0]
            
            test_result = [symmetry, emdid, sz_orig, err_ang1_norefine, 
                           err_ang2_norefine, t_norefine, err_ang1_refine, 
                           err_ang2_refine, t_refine,err_ang1_eman, 
                           err_ang2_eman, t_eman]
                                    
            print(test_result)
            results.append(test_result)

            # Plot FSCs
            # vol_ref = src.read_write.read_mrc(fnames_dict["ref"])
            # vol_eman = src.read_write.read_mrc(fnames_dict["aligned_eman"])
            # vol_norefine = src.read_write.read_mrc(fnames_dict["aligned_norefine"])
            # vol_refine = src.read_write.read_mrc(fnames_dict["aligned_refine"])
            
            # resAa, resAb, resAc, fig = src.fsc.plotFSC3(vol_ref, vol_eman, 
            #                   vol_ref, vol_norefine, 
            #                   vol_ref, vol_refine, 
            #                   labels = ['eman','no refine','refine'],
            #                   cutoff = 0.5, 
            #                   pixelsize=(test_densities[testidx])[2],
            #                   figname = "fsc_{0:s}.png".format(emdid))
            
  
    if not disable_preprocess:
        eman_script.close()
        emalign_script.close()
    

    if not disable_analysis:
        df = pd.DataFrame(results, columns = ['symmetry','emdid','size_orig',
                        'err_ang1_norefine','err_ang2_norefine','t_norefine',
                        'err_ang1_refine','err_ang2_refine','t_refine',
                        'err_ang1_eman', 'err_ang2_eman', 't_eman'])    
        df.to_csv("results_eman.txt")
        df.to_excel("results_eman.xlsx")

#%%
def results_noise():

    disable_preprocess = False
    disable_analysis = True
    
    if not disable_preprocess:
        # Create EMAN script file
        eman_script =  open("run_eman_snr_test.sh", 'w')
        emalign_script =  open("run_emalign_snr_test.sh", 'w')


        #Generate two noiseless density maps
                
        vol = src.read_write.read_mrc('./data/map_2660_ref.mrc')
        # Downsample to size 128 for speedup
        sz = 128
        vol = src.common_finufft.cryo_downsample(vol,[sz,sz,sz])
        # Generate a random rotation
        R = np.squeeze(src.rand_rots.rand_rots(1))
        assert abs(np.linalg.det(R)-1) <1.0e-8
        # Rotate the reference volume by the random rotation
        vol_transformed = src.fastrotate3d.fastrotate3d(vol,R)
        # Add shift of up to 10%
        shift = np.floor((np.random.rand(3)-1/2)*sz*0.1)
        vol_transformed = src.reshift_vol.reshift_vol(vol_transformed,shift)
        # Normalize volumes
        vol = vol/np.linalg.norm(vol.ravel())
        vol_transformed = vol_transformed/np.linalg.norm(vol_transformed.ravel())


    results = []    
    SNR_list = [10000, 1, 1/2, 1/8, 1/32, 1/64, 1/128, 1/256, 1/512, 1/1024]
    
    for testidx in range(len(SNR_list)):
        
        snr = SNR_list[testidx]
        
        # Generate filenames
        working_dir = "./noise_test"   
        fnames_dict={}        
        fnames_dict['ref_noisy'] = os.path.join(working_dir,'map_{0:d}_ref.mrc'.format(testidx))
        fnames_dict['transformed'] = os.path.join(working_dir,'map_{0:d}_transformed.mrc'.format(testidx))
        fnames_dict['aligned_norefine'] = os.path.join(working_dir,'map_{0:d}_aligned_norefine.mrc'.format(testidx))
        fnames_dict['aligned_refine'] = os.path.join(working_dir,'map_{0:d}_aligned_refine.mrc'.format(testidx))
        fnames_dict['aligned_eman'] = os.path.join(working_dir,'map_{0:d}_aligned_eman.mrc'.format(testidx))
        fnames_dict['ref_rot'] = os.path.join(working_dir,'ref_rot_{0:d}.txt'.format(testidx))
        fnames_dict['output_eman'] = os.path.join(working_dir,'params_eman_{0:d}.txt'.format(testidx))        
        fnames_dict['timing_eman'] = os.path.join(working_dir,'timing_eman_{0:d}.txt'.format(testidx))
        fnames_dict['output_norefine'] = os.path.join(working_dir,'output_norefine_{0:d}.txt'.format(testidx))
        fnames_dict['output_refine'] = os.path.join(working_dir,'output_refine_{0:d}.txt'.format(testidx))
        fnames_dict['timing_norefine'] = os.path.join(working_dir,'timing_norefine_{0:d}.txt'.format(testidx))
        fnames_dict['timing_refine'] = os.path.join(working_dir,'timing_refine_{0:d}.txt'.format(testidx))
                
        logger.info('Test %d/%d snr=%5.4f',testidx+1,len(SNR_list),snr)
            
        if not disable_preprocess:
                        
            sigma = np.sqrt(1/snr)
            
            n1 = sigma*np.random.randn(*vol.shape)/np.sqrt(vol.size)
            n2 = sigma*np.random.randn(*vol_transformed.shape)/np.sqrt(vol_transformed.size)
            vol_noisy = vol+n1
            vol_transformed_noisy = vol_transformed+n2
            
            plt.imshow(vol_noisy[64])
            plt.imshow(vol_transformed_noisy[64])
            plt.show()
            
            # Save volumes to align        
            src.read_write.write_mrc(fnames_dict["ref_noisy"], vol_noisy)                
            src.read_write.write_mrc(fnames_dict["transformed"], vol_transformed_noisy)                
    
            # Save rotation paramters    
            with open(fnames_dict["ref_rot"], 'w') as f:
                f.write(str(R)+"\n")
                    
                # Generate EMAN script        
       
            eman_script.write("start=`date +%s`\n")
            eman_script.write(("e2proc3d.py {0:s} {1:s} --alignref {2:s} " + 
                          "--align rotate_translate_3d_tree "+
                          "--verbose 1 > {3:s}\n"
                          ).format(fnames_dict["transformed"],
                                   fnames_dict["aligned_eman"],
                                   fnames_dict["ref_noisy"], fnames_dict["output_eman"]))
            eman_script.write("end=`date +%s`\n")
            eman_script.write(("echo `expr $end - $start` > {0:s}\n\n"
                               ).format(fnames_dict["timing_eman"]))
        
            eman_script.flush()
            
            ###################################################################
            # You have to run the EMAN script run_eman.sh from 
            # command line, It is not possible ti run it from within Pyhton since
            # it uses a different conda environment
            ###################################################################
        
        
            # Generate emalign script
            sz_ds = 64
            n_projs = 30
                    
            # Run alignment without refiment                    
            align_cmd = emalign_cmd + (' --vol1 {0:s} --vol2 {1:s} '+
                '--output-vol {2:s} --downsample {3:d} '+
                ' --n-projs {4:d} ' + 
                '--output-parameters {5:s} --no-refine '+
                '--verbose').format(fnames_dict['ref_noisy'], 
                                    fnames_dict['transformed'],
                                    fnames_dict['aligned_norefine'], 
                                    sz_ds, n_projs,
                                    fnames_dict['output_norefine'])
                                        
            emalign_script.write("echo Running test {0:d} snr={1:5.4f}\n\n".format(testidx,snr))
            emalign_script.write("start=`date +%s`\n")
            emalign_script.write(align_cmd+"\n")
            emalign_script.write("end=`date +%s`\n")
            emalign_script.write(("echo `expr $end - $start` > {0:s}\n"
                                   ).format(fnames_dict["timing_norefine"]))
            emalign_script.write("\n")
       
        
            # Run alignment with refiment
            align_cmd = emalign_cmd + (' --vol1 {0:s} --vol2 {1:s} '+
                '--output-vol {2:s} --downsample {3:d} '+
                ' --n-projs  {4:d} ' + 
                '--output-parameters {5:s} '+
                '--verbose').format(fnames_dict['ref_noisy'], fnames_dict['transformed'],
                                    fnames_dict['aligned_refine'], 
                                    sz_ds, n_projs,
                                    fnames_dict['output_refine'])
                                    
            emalign_script.write("start=`date +%s`\n")
            emalign_script.write(align_cmd+"\n")
            emalign_script.write("end=`date +%s`\n")
            emalign_script.write(("echo `expr $end - $start` > {0:s}\n"
                                  ).format(fnames_dict["timing_refine"]))
            emalign_script.write("\n\n")
        
            emalign_script.flush()
            
        if not disable_analysis:
        # Parse outputs
        
            R_ref = rot_from_params_file(fnames_dict["ref_rot"])        
            
            # Eman accuracy
            with open(fnames_dict["output_eman"], 'r') as f:
                lines = f.readlines() 
                res = (parse.search("'az':{0:g},'alt':{},'phi':{},'tx'",lines[1]))
                az = float(res[0])
                alt = float(res[1])
                phi = float(res[2])
                
                # It seems that 
                # (scipy.spatial.transform.Rotation.from_matrix(R.transpose())).as_euler('zyz',degrees=True)
                # is equivalent to [az,alt,phi-180].
                # In other words, 
                # (scipy.spatial.transform.Rotation.from_euler('zyz',[float(az),float(alt),float(phi)-180], degrees=True)).as_matrix()-R.transpose()
                # is small
                
                R_eman = ((scipy.spatial.transform.Rotation.from_euler('zyz',[float(az),float(alt),float(phi)], degrees=True)).as_matrix()).transpose()
                err_ang1_eman, err_ang2_eman = measure_error(R_ref,R_eman,'C1')
                
            # Eman Timing
            with open(fnames_dict["timing_eman"], 'r') as f:
                t_eman = float(f.readline())
              
            # Emalign accuracy
            try:
                R_norefine = rot_from_params_file(fnames_dict["output_norefine"])
                err_ang1_norefine, err_ang2_norefine = measure_error(R_ref, 
                                                R_norefine.transpose(), 'C1')
            except:
                err_ang1_norefine = -1
                err_ang2_norefine = -1

            try:
                R_refine = rot_from_params_file(fnames_dict["output_refine"])
                err_ang1_refine, err_ang2_refine = measure_error(R_ref, 
                                    R_refine.transpose(), 'C1')
            except:
                err_ang1_refine = -1
                err_ang2_refine = -1

            # Emalign timing
            with open(fnames_dict["timing_norefine"], 'r') as f:
                t_norefine = float(f.readline())
              
            with open(fnames_dict["timing_refine"], 'r') as f:
                t_refine = float(f.readline())
    
            
            # Save results
            #vol_ref = src.read_write.read_mrc(fnames_dict["ref_noisy"])
            #sz_orig = vol_ref.shape[0]
            
            test_result = [snr, err_ang1_norefine, 
                           err_ang2_norefine, t_norefine, err_ang1_refine, 
                           err_ang2_refine, t_refine,err_ang1_eman, 
                           err_ang2_eman, t_eman]
                                    
            print(test_result)
            results.append(test_result)

            # Plot FSCs
            # vol_ref = src.read_write.read_mrc(fnames_dict["ref"])
            # vol_eman = src.read_write.read_mrc(fnames_dict["aligned_eman"])
            # vol_norefine = src.read_write.read_mrc(fnames_dict["aligned_norefine"])
            # vol_refine = src.read_write.read_mrc(fnames_dict["aligned_refine"])
            
            # resAa, resAb, resAc, fig = src.fsc.plotFSC3(vol_ref, vol_eman, 
            #                   vol_ref, vol_norefine, 
            #                   vol_ref, vol_refine, 
            #                   labels = ['eman','no refine','refine'],
            #                   cutoff = 0.5, 
            #                   pixelsize=(test_densities[testidx])[2],
            #                   figname = "fsc_{0:s}.png".format(emdid))
            
  
    if not disable_preprocess:
        eman_script.close()
        emalign_script.close()
    

    if not disable_analysis:
        df = pd.DataFrame(results, columns = ['snr','err_ang1_norefine',
                        'err_ang2_norefine','t_norefine',
                        'err_ang1_refine','err_ang2_refine','t_refine',
                        'err_ang1_eman', 'err_ang2_eman', 't_eman'])    
        df.to_csv("results_snr.txt")
        df.to_excel("results_snr.xlsx")

def test_stability():
    '''
    Test that there are no problems when the rotation is around the z axis 
    (due) to singularity of the Euler angles representation

    Returns
    -------
    None.

    '''
    vol = src.read_write.read_mrc('./data/map_2660_ref.mrc')
    # Downsample to size 128 for speedup
    sz = 128
    vol = src.common_finufft.cryo_downsample(vol,[sz,sz,sz])
    # Generate a random rotation
   
   
    angle = np.pi/3
    R = np.array(
       [[np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [    0,           0,      1]] 
       )
   
    # Rotate the reference volume by the random rotation
    vol_transformed = src.fastrotate3d.fastrotate3d(vol,R)

    class Struct:
        pass

    opt = Struct()
    opt.Nprojs = 3
    opt.downsample = 64
    opt.no_refine = False

    from src.align_volumes_3d import align_volumes
    bestR, bestdx, reflect, vol2aligned, bestcorr = align_volumes(vol, vol_transformed, verbose = 1 , opt = opt)
    print(np.linalg.norm(bestR.transpose()-R))  #Should be about 1.0e-3


## Main part: run tests
#download_data('./data')

#results_varying_downsampling()
#results_varying_Nprojs()
results_comparison_to_eman()
#results_noise()

#test_stability()
