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

import src.cryo_fetch_emdID
import src.rand_rots
import src.read_write
import src.common_finufft
import src.fastrotate3d
import src.reshift_vol
import src.SymmetryGroups

test_densities = [
    ['C1',     '2660',     3.2],
    ['C2',     '0667',    6.2],
    ['C3',     '0731',    2.85],
    ['C4',     '0882',    3.3],
    ['C5',    '21376',    2.6],
    ['C7',    '11516',    2.38],
    ['C8',    '21143',    3.63],
    ['C11',  '6458',    4.7],
    ['D2',    '30913',    1.93],
    ['D3',    '20016',    2.77],
    ['D4',    '22462',    2.06],
    ['D7',     '9233',    2.1],
    ['D11',    '21140',    3.68],
    ['T',     '4179',    4.1],
    ['O',    '22658',    1.36],
    ['I',   '24494',    2.27]
]

working_dir = "./data"

emalign_cmd = "/home/yoelsh/.local/bin/emalign"

# Setup logger
logging.basicConfig(level=logging.DEBUG,
format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()


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
        2600168882, 1256372140,  757182475, 3440989097, 2686223822,
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
       aligned_refine: 'map_EMDID_aligned_refine.mrc',
       aligned_eman: 'map_EMDID_aligned_eman.mrc'
    '''
    
    dict={}
    dict['ref'] = os.path.join(working_dir,'map_{0:s}_ref.mrc'.format(emdid))
    dict['transformed'] = os.path.join(working_dir,'map_{0:s}_transformed.mrc'.format(emdid))
    dict['aligned_norefine'] = os.path.join(working_dir,'map_{0:s}_aligned_norefine.mrc'.format(emdid))
    dict['aligned_refine'] = os.path.join(working_dir,'map_{0:s}_aligned_refine.mrc'.format(emdid))
    dict['aligned_eman'] = os.path.join(working_dir,'map_{0:s}_aligned_eman.mrc'.format(emdid))
    
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
    axis_ref = tmp_rotvec/np.linalg.norm(tmp_rotvec);
    angle_ref = np.linalg.norm(tmp_rotvec);


    tmp_R = scipy.spatial.transform.Rotation.from_matrix(g_est @ Rest.transpose())
    tmp_rotvec = tmp_R.as_rotvec()            
    axis_est = tmp_rotvec/np.linalg.norm(tmp_rotvec);
    angle_est = np.linalg.norm(tmp_rotvec);


    err_ang1 = math.acos(np.dot(axis_ref,axis_est))/math.pi*180
    err_ang2 = abs(angle_ref - angle_est)/math.pi*180

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
    #subprocess.run(align_cmd)  
    
    # Run with printouts
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    # Grab stdout line by line as it becomes available.  
    # This will loop until p terminates.
    while p.poll() is None:
        l = p.stdout.readline() # This blocks until it receives a newline.
        print(l)
        # When the subprocess terminates there might be unconsumed output 
        # that still needs to be processed.
        print(p.stdout.read())
        
    t_end = time.time()
    return t_end-t_start


def results_varying_N():
    init_random_state()
    
    
    results = []    
    sizes = [16, 32, 64, 128]  # Sizes of downsampled volumes
        
    for sz_ds in sizes:
    #sz_ds = 64 # Size of downsampled volume
    
        for testidx in range(len(test_densities)):
                              
            #Generate two density maps
        
            test_data = test_densities[testidx]
            symmetry = test_data[0]
            emdid = test_data[1]        
            fnames_dict = get_test_filenames(working_dir, emdid)
    
            logger.info('Test %d/%d %s  (EMD%s)',testidx+1,len(test_densities),symmetry,emdid)
            
            vol = src.read_write.read_mrc(fnames_dict['ref'])
            sz_orig = vol.shape[0]  # Size of original volume (before downsampling)
            vol = src.common_finufft.cryo_downsample(vol,[sz_ds,sz_ds,sz_ds])
    
            # Generate a random rotation
            R = np.squeeze(src.rand_rots.rand_rots(1))
            assert abs(np.linalg.det(R)-1) <1.0e-8
    
            # Rotate the reference volume by the random rotation
            volRotated1 = src.fastrotate3d.fastrotate3d(vol,R)
            
            # Add shift
            # volRotated1 = src.reshift_vol.reshift_vol(volRotated1,[-5, 0, 0])
    
            try:
                
                # Save the two volues to align
                vol_ref_name = str(uuid.uuid4())+'.mrc'
                vol_ref_name = os.path.join(working_dir, vol_ref_name)
                src.read_write.write_mrc(vol_ref_name, vol)
    
                vol_rot_name = str(uuid.uuid4())+'.mrc'
                vol_rot_name = os.path.join(working_dir, vol_rot_name)
                src.read_write.write_mrc(vol_rot_name, volRotated1)
                
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
                '--output-parameters {4:s} --no-refine '+
                '--verbose').format(vol_ref_name, vol_rot_name,
                                    vol_aligned_norefine_name, sz_ds, 
                                    params_norefine_name)                         
           
                                    
                # Run alignment command
                t_norefine = run_cmd(align_cmd)
    
                # Read estimated matrix from paramters file
                Rest = rot_from_params_file(params_norefine_name)
                Rest = Rest.transpose()
                
                # Calculate error between ground-truth and estimated rotation
                err_ang1_norefine, err_ang2_norefine = measure_error(R, Rest, symmetry)
    
    
                ####
                # Run alignment with refiment
                ####
                align_cmd = emalign_cmd + (' --vol1 {0:s} --vol2 {1:s} '+
                '--output-vol {2:s} --downsample {3:d} '+
                '--output-parameters {4:s}  '+
                '--verbose').format(vol_ref_name, vol_rot_name, 
                                    vol_aligned_refine_name, sz_ds, 
                                    params_refine_name)                                
                
                # Run alignment command
                t_refine = run_cmd(align_cmd)
    
                # Read estimated matrix from paramters file
                Rest = rot_from_params_file(params_refine_name)
                Rest = Rest.transpose()
                
                # Calculate error between ground-truth and estimated rotation
                err_ang1_refine, err_ang2_refine = measure_error(R, Rest, symmetry)
    
                    
                test_result = [symmetry, emdid, sz_orig, sz_ds,
                               err_ang1_norefine, err_ang2_norefine, t_norefine,
                                   err_ang1_refine, err_ang2_refine, t_refine]
                    
                print(test_result)
                results.append(test_result)
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

    return results

# def temp():            
#     np.random.set_state(random_state)

# # Set table of tested maps.
# # First column is symmetry type, second column is EMD-id, third column is EMDB 
# # reported resolution


#     for testidx in range(len(test_densities)):
    
#         #Generate two density maps
    
#         test_data = test_densities[testidx]
#         symmetry = test_data[0]
#         emdid = test_data[1]
#         resolution = test_data[2]

#         logger.info('Test %d/%d %s  (EMD%s)',testidx+1,len(test_densities),symmetry,emdid)

#         try:
#             mapfile = next(tempfile._get_candidate_names())        
#             src.cryo_fetch_emdID.cryo_fetch_emdID(emdid,mapfile)
#             vol = src.read_write.read_mrc(mapfile)
#             sz_ds = 64;
#             vol = src.common_finufft.cryo_downsample(vol,[sz_ds,sz_ds,sz_ds])
#             #vol = cryo_downsample(vol,sz_ds,0);
#         finally:
#             if os.path.exists(mapfile):
#                 os.remove(mapfile)
                        
#         # Generate a random rotation
#         R, dummy = np.linalg.qr(np.random.rand(3,3))

#         # Rotate the reference volume by the random rotation
#         volRotated1 = src.fastrotate3d.fastrotate3d(vol,R)
   
#         # Generate a reflected volume
#         volRotated2 = np.flip(volRotated1,axis=2)
#         volRotated2 = src.reshift_vol.reshift_vol(volRotated2,[-5, 0, 0])
   
#         # Add shift
#         volRotated1 = src.reshift_vol.reshift_vol(volRotated1,[-5, 0, 0])
        
#download_data('./data')

results = results_varying_N()
print(results)
import pandas as pd
df = pd.DataFrame(results, columns = ['symmetry','emdid','size_orig', 'size_ds','err_ang1_norefine','err_ang2_norefine','t_norefine','err_ang1_refine','err_ang2_refine','t_refine'])
df.to_csv("results.txt")
df.to_excel("results.xlsx")
#df.loc[df['symmetry']=='C1','err_ang1']