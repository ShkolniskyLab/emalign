# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 22:51:15 2022

@author: yoel
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_excel("results.xlsx")

#%%
# Figure 1:
#    
# Plot accuracy vs downsampling size, averaged over all symmetry groups
# 

err1 = []
err2 = []
ds_sizes = [16, 32, 64, 128]
for sz in ds_sizes:
    err1.append(df.loc[df['size_ds']==sz,'err_ang1_norefine'].tolist())
    err2.append(df.loc[df['size_ds']==sz,'err_ang2_norefine'].tolist())
    
    
plt.figure()    
plt.subplot(1,2,1)
plt.boxplot(err1)
plt.xticks(range(1,len(ds_sizes)+1), ds_sizes)
plt.minorticks_on()
plt.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.4)
plt.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.2)
plt.subplot(1,2,2)
plt.boxplot(err2)
plt.xticks(range(1,len(ds_sizes)+1), ds_sizes)
plt.minorticks_on()
plt.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.4)
plt.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.2)
plt.show()


#%%
# Figure 2:
# Same as Fig.1, but focus only on sizes 64 and 128
err1 = []
err2 = []
ds_sizes = [64, 128]
for sz in ds_sizes:
    err1.append(df.loc[df['size_ds']==sz,'err_ang1_norefine'].tolist())
    err2.append(df.loc[df['size_ds']==sz,'err_ang2_norefine'].tolist())
    
    
plt.figure()    
plt.subplot(1,2,1)
plt.boxplot(err1)
plt.xticks(range(1,len(ds_sizes)+1), ds_sizes)
plt.minorticks_on()
plt.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.4)
plt.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.2)
plt.subplot(1,2,2)
plt.boxplot(err2)
plt.xticks(range(1,len(ds_sizes)+1), ds_sizes)
plt.minorticks_on()
plt.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.4)
plt.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.2)
plt.show()

#%%
# Figure 3:
# For downsampling size 64, show error before and after refinement
err_norefine = []
err_refine = []
ds_size = 64

df1 = df.loc[df['size_ds']==64, 
            ['err_ang1_norefine', 'err_ang2_norefine', 
             'err_ang1_refine', 'err_ang2_refine']]
    
err_norefine = [df1["err_ang1_norefine"].tolist(), df1["err_ang2_norefine"].tolist()]
err_refine = [df1["err_ang1_refine"].tolist(), df1["err_ang2_refine"].tolist()]

    
plt.figure()    
plt.subplot(1,2,1)
plt.boxplot(err_norefine)
plt.xticks([0, 1, 2],['', 'err1', 'err2'])
plt.minorticks_on()
plt.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.4)
plt.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.2)
plt.subplot(1,2,2)
plt.boxplot(err_refine)
plt.xticks([0, 1, 2],['', 'err1', 'err2'])
plt.minorticks_on()
plt.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.4)
plt.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.2)
plt.show()

#%%
# Figure 4:
# Show timing in seconds vs volume size

ds_sizes = [64, 128]

# Time without refinement
for sz in ds_sizes:
    unsorted_series = df.loc[df['size_ds']==sz,['size_orig','t_norefine']]
    sorted_series = unsorted_series.sort_values(by=['size_orig'])
    plt.plot(sorted_series["size_orig"],sorted_series["t_norefine"])
    
# Time with refinement
for sz in ds_sizes:
    unsorted_series = df.loc[df['size_ds']==sz,['size_orig','t_refine']]
    sorted_series = unsorted_series.sort_values(by=['size_orig'])
    plt.plot(sorted_series["size_orig"],sorted_series["t_refine"])
    
plt.minorticks_on()
plt.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.4)
plt.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.2)    
plt.legend(["64 NR", "128 NR", "64 R", "128 R"])
plt.savefig('XXX.png', dpi=300)
plt.show()

    

#%%
# Figure 5:
# Error of eman vs refined emalign

df = pd.read_excel("results_eman.xlsx")

barWidth = 15
 
err_refine = (df["err_ang1_refine"]+df["err_ang2_refine"]).tolist()
err_eman = (df["err_ang1_eman"]+df["err_ang2_eman"]).tolist()
 
# Set position of bar on X axis
br1 = 100*np.arange(len(err_norefine))
br2 = [x + barWidth for x in br1]
 
# Make the plot
plt.bar(br1, err_refine, fill='False', hatch='///', width = barWidth,
        edgecolor ='grey', label ='emalign')
plt.bar(br2, err_eman, fill='False', hatch='...', width = barWidth,
        edgecolor ='grey', label ='eman')

# Adding Xticks
#plt.xlabel('Dataset', fontweight ='bold', fontsize = 15)
plt.ylabel('Error', fontsize = 8)
plt.xticks([r + barWidth for r in br1.ravel()],
        ["{0:04d}".format(x) for x in df["emdid"]], fontsize = 6)
 
plt.legend()
plt.savefig('XXX.png', dpi=300)
plt.show()


#%%
# Figure 6:
# Timing fined emalign

df = pd.read_excel("results_eman.xlsx")

barWidth = 15
 
timing_refine = (df["t_refine"]).tolist()
timing_eman = (df["t_eman"]).tolist()
 
# Set position of bar on X axis
br1 = 100*np.arange(len(err_norefine))
br2 = [x + barWidth for x in br1]
 
# Make the plot
plt.bar(br1, timing_refine, fill='False', hatch='///', width = barWidth,
        edgecolor ='grey', label ='emalign')
plt.bar(br2, timing_eman, fill='False', hatch='...', width = barWidth,
        edgecolor ='grey', label ='eman')

# Adding Xticks
#plt.xlabel('Dataset', fontweight ='bold', fontsize = 15)
plt.ylabel('Time (sec)', fontsize = 8)
plt.xticks([r + barWidth for r in br1.ravel()],
        ["{0:04d}".format(x) for x in df["emdid"]], fontsize = 6)
 
plt.legend()
plt.savefig('XXX.png', dpi=300)
plt.show()


#%%
# Delete:
# x = []
# y_norefine = []
# yerr_norefine = []
# y_refine = []
# yerr_refine = []

# ds_sizes = [32, 64, 128]
# for sz in ds_sizes:
#     x.append(sz)
    
#     d_norefine = df.loc[df['size_ds']==sz,'err_ang1_norefine'].tolist()    
#     y_norefine.append(np.mean(d_norefine))
#     yerr_norefine.append(np.std(d_norefine))
    
#     d_refine = df.loc[df['size_ds']==sz,'err_ang1_refine'].tolist()    
#     y_refine.append(np.mean(d_refine))
#     yerr_refine.append(np.std(d_refine))


# plt.figure()
# plt.errorbar(x, y_norefine, yerr_norefine)
# plt.errorbar(x, y_refine, yerr_refine)
