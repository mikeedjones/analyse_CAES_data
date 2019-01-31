# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:56:31 2019

@author: Michael
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.optimize as so
import scipy.ndimage as sim
import scipy.ndimage.measurements as scim
import scipy.stats as scis
import scipy.constants as sc
import operator
import os
import pdb
import bias as b
from tqdm import tqdm
import re
import analyse_electron_spot as aes

pixpermm=53.4

def show_6(groups,shot_no=True,fout='x-section_fits.pdf'):
    
    if shot_no:
        shot_no=[]
        N=np.linspace(0,3500,6)
        
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]
        
        for n in N:
            shot_no.append(find_nearest(list(groups.keys()),n))
    
    fig, ax=plt.subplots(3,2)
    
    axes=ax.reshape(6,1)
    labs=[True,False,True,False,True,False]
    
    for ax_c,shot,lab in zip(axes,shot_no,labs):
        groups[shot].save_fig(ax=ax_c[0],ylab=lab)
#        ax_c[0].imshow(groups[shot].crop)
        ax_c[0].set_title('{:.0f} V'.format(np.round(groups[shot].bias,decimals=-1)))
    plt.tight_layout()
    plt.savefig(fout,dpi=300,bbox='tight')

def fit_low_e(shot,flag=False):
    
    if flag:
        biases, fits = [[] for _ in range(2)]
        for key,shot in tqdm(shot.items()):
            bias,fit_stdx=fit_low_e(shot)
            biases.append(bias)
            fits.append(fit_stdx/pixpermm)
        
        return biases, fits
    
    shot.single_shot(0)
    shot.fit_gaussain(0)
    
    return shot.bias, shot.stdy

#    plt.savefig('fitted_gaussians.pdf',bbox='tight',dpi=300)

def fit_shot(shot,flag=False):
    
    if flag:
        biases, fits, std = [[] for _ in range(3)]
        for key,shot in tqdm(shot.items()):
            bias,fit_stdx,loc_stdx,=fit_shot(shot)
            biases.append(bias)
            fits.append(fit_stdx/pixpermm)
            std.append(loc_stdx/pixpermm)
        
        return biases, fits, std
    
    shot.single_shot(0)
    shot.fit_gaussain(0)
    
    return shot.bias, shot.stdy, shot.loc_group.stdx

def show_fake(shot):
    fig,ax=plt.subplots(3,2,sharex='row',sharey='row')

    shot.single_shot(0)
    
    shot.fit_gaussain(0)
    shot.save_fig(ax=ax[0][1],ylab=False)
    shot.transverse(ax=[ax[1][1],ax[2][1]],ylab=False)
    ax[0][1].set_title('Added noise')
    
    shot.loc_group.save_fig(ax=ax[0][0],ylab=True)   
    shot.loc_group.locs_trans(ax[1][0])
    shot.loc_group.locs_trans(ax[2][0])
    shot.loc_group.transverse(ax=[ax[1][0],ax[2][0]],ylab=True)
    ax[0][0].set_title('Underlying')
    
    plt.tight_layout()
    plt.show()
#    plt.savefig('fitted_gaussians.pdf',bbox='tight',dpi=300)
    
def importims_locs(dir_ims, dir_locs, verbose=0, biases=[], outdir='out', outfile='none_given',ptype='e',decimation=1):
    if not os.path.exists (dir_ims):
        print("No such folder")
        return -1
    
        
    bias_groups={}
    bg_open=False
    try:
        for root_ims, dirs, files_ims in os.walk(dir_ims+'/../'):
            for names in files_ims:
                if "bg" in names:
                    filepath = os.path.join(root_ims,names)
                    if bg_open == False:
                        bg=b.bias_group(0, 0, np.transpose(np.genfromtxt(filepath))[5:,5:])#[10:][10:])
                        bg_open=True
                        continue
                    else:
                        bg.add_shot(0, 0, np.transpose(np.genfromtxt(filepath))[5:,5:])#[10:][10:])
                        continue
                else: bg=0
    except:
        print(root_ims)
        import traceback
        # Print the stack traceback
        traceback.print_exc()
    
    locs_dict={}
    
    try:
        for root_locs, dirs_locs, files_locs in os.walk(dir_locs):
            for names in tqdm(files_locs[0::decimation],smoothing=0):
                if names==".DS_Store" : continue   
                if "header" in names : continue    
                if "bg" in names : continue
                if "profile.mp4" in names: continue
                
                filepath = os.path.join(root_locs,names)
                
                locs_dict[int(re.split('_|\.',names)[-2])]=np.load(filepath)            
            
                
    except:
        print(root_ims)
        import traceback
        # Print the stack traceback
        traceback.print_exc()
    
    try:
        for root_ims, dirs_ims, files_ims in os.walk(dir_ims):
            for names in files_ims:
                filepath = os.path.join(root_ims,names)
                if "header" in names:
                    header=np.genfromtxt(filepath,names=True)
    

            for names in tqdm(files_ims[0::decimation],smoothing=0):
                if names==".DS_Store" : continue   
                if "header" in names : continue    
                if "bg" in names : continue
                if "profile.mp4" in names: continue
            
                filepath = os.path.join(root_ims,names)
                
                shot_no=int(re.split('_|\.',names)[-2])
            
                header_line=shot_no
                if header_line>len(header["V4"]): continue
            
                if ptype=='i':
                    bias=header["V5"][header_line]#shot numbering starts from 1
                bias=header["V4"][header_line]
                
                rounding=0
                group=shot_no
                if 0<bias<5000 and header['voltage_change'][header_line]==0:
                    if np.around(group,decimals=rounding) in bias_groups:
                        bias_groups[np.around(group,decimals=rounding)].add_shot(header['voltage_change'][header_line], bias, np.transpose(np.genfromtxt(filepath))[5:,5:])#,z0=header['z0'][shot_no-1],errz0=header['errz0'][shot_no-1])#,modelstdx=header['stdx'][shot_no-1])#[10:][10:])
                    else:
                        bias_groups[np.around(group,decimals=rounding)]=b.bias_group(header['voltage_change'][header_line], bias, np.transpose(np.genfromtxt(filepath))[5:,5:])#,z0=header['z0'][shot_no-1],errz0=header['errz0'][shot_no-1])#,modelstdx=header['stdx'][shot_no-1])#[10:][10:])
                    bias_groups[np.around(group,decimals=rounding)].add_locs(locs_dict[np.around(group,decimals=rounding)])
        return bias_groups, bg, bg_open
    
    except:
        import traceback
        # Print the stack traceback
        traceback.print_exc()
        print(group)
        

def importims(dir_ims, verbose=0, biases=[], outdir='out', outfile='none_given',ptype='e',decimation=1):
    if not os.path.exists (dir_ims):
        print("No such folder")
        return -1
    
        
    bias_groups={}
    bg_open=False
    try:
        for root_ims, dirs, files_ims in os.walk(dir_ims+'/../'):
            for names in files_ims:
                if "bg" in names:
                    filepath = os.path.join(root_ims,names)
                    if bg_open == False:
                        bg=b.bias_group(0, 0, np.transpose(np.genfromtxt(filepath))[5:,5:])#[10:][10:])
                        bg_open=True
                        continue
                    else:
                        bg.add_shot(0, 0, np.transpose(np.genfromtxt(filepath))[5:,5:])#[10:][10:])
                        continue
                else: bg=0
    except:
        print(root_ims)
        import traceback
        # Print the stack traceback
        traceback.print_exc()
    
    try:
        for root_ims, dirs_ims, files_ims in os.walk(dir_ims):
            for names in files_ims:
                filepath = os.path.join(root_ims,names)
                if "header" in names:
                    header=np.genfromtxt(filepath,names=True)
    

            for names in tqdm(files_ims[0::decimation],smoothing=0):
                if names==".DS_Store" : continue   
                if "header" in names : continue    
                if "bg" in names : continue
                if "profile.mp4" in names: continue
            
                filepath = os.path.join(root_ims,names)
                
                shot_no=int(re.split('_|\.',names)[-2])
            
                header_line=shot_no
                if header_line>len(header["V4"]): continue
            
                if ptype=='i':
                    bias=header["V5"][header_line]#shot numbering starts from 1
                bias=header["V4"][header_line]
                
                rounding=0
                group=shot_no
                if 0<bias<5000 and header['voltage_change'][header_line]==0:
                    if np.around(group,decimals=rounding) in bias_groups:
                        bias_groups[np.around(group,decimals=rounding)].add_shot(header['voltage_change'][header_line], bias, np.transpose(np.genfromtxt(filepath))[5:,5:])#,z0=header['z0'][shot_no-1],errz0=header['errz0'][shot_no-1])#,modelstdx=header['stdx'][shot_no-1])#[10:][10:])
                    else:
                        bias_groups[np.around(group,decimals=rounding)]=b.bias_group(header['voltage_change'][header_line], bias, np.transpose(np.genfromtxt(filepath))[5:,5:])#,z0=header['z0'][shot_no-1],errz0=header['errz0'][shot_no-1])#,modelstdx=header['stdx'][shot_no-1])#[10:][10:])
        return bias_groups, bg, bg_open
    
    except:
        import traceback
        # Print the stack traceback
        traceback.print_exc()
        print(group)