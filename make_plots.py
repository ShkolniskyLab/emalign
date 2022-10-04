# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 22:51:15 2022

@author: yoel
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


figs_dir = './figs/'

def make_full_figname(figname):
    return os.path.join(figs_dir, figname)

#%%
# Figure 1:
#    
# Plot accuracy vs downsampling size, averaged over all symmetry groups
# 

df = pd.read_excel("results_varying_downsampling.xlsx")
err1 = []
err2 = []
ds_sizes = [16, 32, 64, 128]
for sz in ds_sizes:
    err1.append(df.loc[df['size_ds']==sz,'err_ang1_norefine'].tolist())
    err2.append(df.loc[df['size_ds']==sz,'err_ang2_norefine'].tolist())
    
    
plt.rcParams['text.usetex'] = True    
plt.figure()    
plt.subplot(1,2,1)
plt.boxplot(err1)
plt.xticks(range(1,len(ds_sizes)+1), ds_sizes)
plt.minorticks_on()
plt.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.4)
plt.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.2)
plt.ylabel('$e_{1}$ (degrees)', fontsize = 8)
plt.subplot(1,2,2)
plt.boxplot(err2)
plt.xticks(range(1,len(ds_sizes)+1), ds_sizes)
plt.minorticks_on()
plt.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.4)
plt.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.2)
plt.ylabel('$e_{2}$ (degrees)', fontsize = 8)
plt.subplots_adjust(wspace=0.4)
plt.savefig(make_full_figname("downsampling_accuracy_norefine.png"), dpi=300)
plt.show()


#%%
# Figure 2:
# Same as Fig.1, but focus only on sizes 64 and 128

df = pd.read_excel("results_varying_downsampling.xlsx")
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
plt.ylabel('$e_{1}$ (degrees)', fontsize = 8)
plt.subplot(1,2,2)
plt.boxplot(err2)
plt.xticks(range(1,len(ds_sizes)+1), ds_sizes)
plt.minorticks_on()
plt.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.4)
plt.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.2)
plt.ylabel('$e_{2}$ (degrees)', fontsize = 8)
plt.subplots_adjust(wspace=0.4)
plt.savefig(make_full_figname("downsampling_accuracy_norefine_focused.png"), dpi=300)
plt.show()

#%%
# Figure 3:
# For downsampling size 64, show error before and after refinement

df = pd.read_excel("results_varying_downsampling.xlsx")
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
plt.xticks([1, 2],['$e_{1}$', '$e_{2}$'])
plt.minorticks_on()
plt.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.4)
plt.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.2)
plt.ylabel('Error (degrees)', fontsize = 8)
plt.subplot(1,2,2)
plt.boxplot(err_refine)
plt.xticks([1, 2],['$e_{1}$', '$e_{2}$'])
plt.minorticks_on()
plt.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.4)
plt.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.2)
plt.ylabel('Error (degrees)', fontsize = 8)
plt.subplots_adjust(wspace=0.4)
plt.savefig(make_full_figname("downsampling_accuracy_refine.png"), dpi=300)
plt.show()

#%%
# Figure 4:
# Show timing in seconds vs volume size

df = pd.read_excel("results_varying_downsampling.xlsx")
ds_sizes = [64, 128]
markers = ['o', 's', 'v', '+', '.', 'x', 'd', '.']
marker_idx = 0

# Time without refinement
for sz in ds_sizes:
    unsorted_series = df.loc[df['size_ds']==sz,['size_orig','t_norefine']]
    sorted_series = unsorted_series.sort_values(by=['size_orig'])
    plt.plot(sorted_series["size_orig"],sorted_series["t_norefine"],
             label=("{0:d} "+"NR").format(sz), 
             marker = markers[marker_idx])
    marker_idx = (marker_idx + 1) % len(markers)
    
# Time with refinement
for sz in ds_sizes:    
    unsorted_series = df.loc[df['size_ds']==sz,['size_orig','t_refine']]
    sorted_series = unsorted_series.sort_values(by=['size_orig'])
    plt.plot(sorted_series["size_orig"],sorted_series["t_refine"],
             label=("{0:d} "+"R").format(sz), 
             marker = markers[marker_idx])
    marker_idx = (marker_idx + 1) % len(markers)
    
plt.minorticks_on()
plt.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.4)
plt.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.2)    
plt.xlabel("Volume size (voxels)")
plt.ylabel("Time (secs)")

plt.legend()
plt.subplots_adjust(wspace=0.4)
plt.savefig(make_full_figname("downsampling_timing.png"), dpi=300)
plt.show()

    

#%%
# Figure 5:
# Error va n_projs.
# In this plot the error is the sum of err1 and err2

df = pd.read_excel("results_varying_nprojs.xlsx")

err_norefine = []
err_refine = []
n_projs_list = np.sort(np.unique(df["n_projs"]))
for n_projs in n_projs_list:
    data = df.loc[df['n_projs']==n_projs,
                  ['err_ang1_norefine', 'err_ang2_norefine']]
    err_norefine.append((data["err_ang1_norefine"] + data["err_ang2_norefine"]).tolist())

    data = df.loc[df['n_projs']==n_projs,
                  ['err_ang1_refine', 'err_ang2_refine']]
    err_refine.append((data["err_ang1_refine"] + data["err_ang2_refine"]).tolist())

    #err1.append(df.loc[df['n_projs']==n_projs,'err_ang1_norefine'].tolist())
    #err2.append(df.loc[df['n_projs']==n_projs,'err_ang2_norefine'].tolist())
    
    
plt.figure()    
plt.subplot(1,2,1)
plt.boxplot(err_norefine)
plt.xticks(range(1,len(n_projs_list)+1), n_projs_list)
plt.minorticks_on()
plt.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.4)
plt.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.2)
plt.ylabel('$e_{1}+e_{2}$ (degrees)', fontsize = 8)
plt.subplot(1,2,2)
plt.boxplot(err_refine)
plt.xticks(range(1,len(n_projs_list)+1), n_projs_list)
plt.minorticks_on()
plt.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.4)
plt.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.2)
plt.ylabel('$e_{1}+e_{2}$ (degrees)', fontsize = 8)

plt.subplots_adjust(wspace=0.4)
plt.savefig(make_full_figname("nprojs_accuracy.png"), dpi=300)
plt.show()



#%%
# Figure 6
# Show timing vs n_projs with and without refinement

df = pd.read_excel("results_varying_nprojs.xlsx")
n_projs_list = [10, 15, 20, 25, 30, 100]
markers = ['o', 's', 'p', 'h', 'v', 'x', 'd', '.']
marker_idx = 0

# Time without refinement
for n_projs in n_projs_list:
    unsorted_series = df.loc[df['n_projs']==n_projs,['size_orig','t_norefine']]
    sorted_series = unsorted_series.sort_values(by=['size_orig'])
    plt.plot(sorted_series["size_orig"],sorted_series["t_norefine"], 
             label=("{0:d} "+"NR").format(n_projs), 
             marker = markers[marker_idx])
    marker_idx = (marker_idx + 1) % len(markers)
    
# Time with refinement
for n_projs in n_projs_list:
    unsorted_series = df.loc[df['n_projs']==n_projs,['size_orig','t_refine']]
    sorted_series = unsorted_series.sort_values(by=['size_orig'])
    plt.plot(sorted_series["size_orig"],sorted_series["t_refine"],
             label=("{0:d} "+"R").format(n_projs), 
             marker = markers[marker_idx])
    marker_idx = (marker_idx + 1) % len(markers)    
    
plt.minorticks_on()
plt.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.4)
plt.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.2)    
plt.xlabel("Volume size (voxels)")
plt.ylabel("Time (secs)")

plt.legend()
plt.subplots_adjust(wspace=0.4)
plt.savefig(make_full_figname("nprojs_refine.png"), dpi=300)
plt.show()



#%%
# Figure 7:
# Error of eman vs refined emalign

df = pd.read_excel("results_eman.xlsx")

barWidth = 15
 
err_refine = (df["err_ang1_refine"]+df["err_ang2_refine"]).tolist()
err_eman = (df["err_ang1_eman"]+df["err_ang2_eman"]).tolist()
 
# Set position of bar on X axis
br1 = 100*np.arange(len(err_refine))
br2 = [x + barWidth for x in br1]
 
# Make the plot
plt.bar(br1, err_refine, fill='False', hatch='///', width = barWidth,
        edgecolor ='grey', label ='emalign')
plt.bar(br2, err_eman, fill='False', hatch='...', width = barWidth,
        edgecolor ='grey', label ='eman')

# Adding Xticks
#plt.xlabel('Dataset', fontweight ='bold', fontsize = 15)
plt.xlabel("EMDID")
plt.ylabel('$e_{1}+e_{2}$ (degrees)', fontsize = 8)
plt.xticks([r + barWidth for r in br1.ravel()],
        ["{0:04d}".format(x) for x in df["emdid"]], fontsize = 6)
 
plt.legend()
plt.subplots_adjust(wspace=0.4)
plt.savefig(make_full_figname('eman_comparison_accuracy.png'), dpi=300)
plt.show()


#%%
# Figure 8:
# Timing of eman vs emalign with refinement

df = pd.read_excel("results_eman.xlsx")

barWidth = 15
 
timing_refine = (df["t_refine"]).tolist()
timing_eman = (df["t_eman"]).tolist()
 
# Set position of bar on X axis
br1 = 100*np.arange(len(timing_refine))
br2 = [x + barWidth for x in br1]
 
# Make the plot
plt.bar(br1, timing_refine, fill='False', hatch='///', width = barWidth,
        edgecolor ='grey', label ='emalign')
plt.bar(br2, timing_eman, fill='False', hatch='...', width = barWidth,
        edgecolor ='grey', label ='eman')

# Adding Xticks
#plt.xlabel('Dataset', fontweight ='bold', fontsize = 15)
plt.xlabel("EMDID")
plt.ylabel('Time (sec)', fontsize = 8)
plt.xticks([r + barWidth for r in br1.ravel()],
        ["{0:04d}".format(x) for x in df["emdid"]], fontsize = 6)
 
plt.legend()
plt.savefig(make_full_figname('eman_comparison_timing.png'), dpi=300)
plt.show()

