# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 07:43:06 2016

@author: Michael
"""
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import scipy.optimize as so
import scipy.ndimage.measurements as scim
from scipy.ndimage.filters import gaussian_filter
import scipy.interpolate as si
import scipy.constants as sc
import operator
from bias import bias_group

xlimit=145
ylimit=145
pixpermm=53.4*1000

def makefitsize(biases,stdy, color='b',label='None'):
    fitsize=so.curve_fit(sizefit,biases,np.array(stdy)/pixpermm,p0=0.01,bounds=(0,1))
    plt.gcf()
    plt.plot(biases,sizefit(biases,fitsize[0]),color=color)
    plt.xlabel("Einzel lens bias / V")
    plt.ylabel("Electron bunch size / mm")
    
    return fitsize;
    
def makefitT(biases,stdy, color='b',label='None'):
    fitT=so.curve_fit(tempfit,biases,np.array(stdy)/pixpermm,p0=[29,1E-5,0],bounds=(-1,50))
    plt.gcf()
    #plt.plot(biases,np.array(stdy)/pixpermm,color=color,marker='.',linestyle='None',label=label+', {:} K, '.format(fitT[0][0]))#+ '{:.1f} $\mu$m, '.format(fitT[0][1]*1E6))
    fittedfunction=tempfit(biases,fitT[0][0],fitT[0][1],fitT[0][2])#,fitT[0][2])
    plt.plot(biases,fittedfunction,color=color,label=label+', {:.1f} K, '.format(fitT[0][0])+ '{:.1f} $\mu$m, '.format(fitT[0][1]*1E6))
    plt.xlabel("Einzel lens bias / V")
    plt.ylabel("Electron bunch size / mm")
    #print(biases[np.argmin(fittedfunction)])
    #print(biases[np.argmin(stdy)])
    
    return fitT;

A_tab=np.genfromtxt('../../experimental_data/07-03-18/482/A_and_B/A_vs_z_vs_BE.txt')
B_tab=np.genfromtxt('../../experimental_data/07-03-18/482/A_and_B/B_vs_z_vs_BE.txt')
#BE_table=np.genfromtxt('19-08-17/z_v_Ext_v_BE.txt')

def B(E,z):
    e_lens=np.array(B_tab[1:-1,0])

    zarr=np.array(B_tab[0,1:])
     
    B=np.array(B_tab[1:-1,1:])
    return si.RectBivariateSpline(e_lens,zarr,B).ev(E,z)

def A(E,z):
    e_lens=np.array(A_tab[1:-1,0])

    zarr=np.array(A_tab[0,1:])
    
    A=np.array(A_tab[1:-1,1:])
    
    return si.RectBivariateSpline(e_lens,zarr,A).ev(E,z)

def A_blur(E,z,sigmaz):
    Ablur=0
    N=500
    #sigmaz=3E-3
    for blur_z in np.random.normal(loc=0, scale=abs(sigmaz), size=N):
        Ablur=A(E,z+blur_z)+Ablur
 #   Ablur=A(E,z)   
    return Ablur/N
    
def B_blur(E,z,sigmaz):
    Bblur=0
    N=500
    #sigmaz=3E-3
    for blur_z in np.random.normal(loc=0, scale=abs(sigmaz), size=N):
        Bblur=B(E,z+blur_z)+Bblur
#    Bblur=B(E,z)
    return Bblur/N

def tempfit(U, T, sigmay, z):#, yfwhm):
    U=np.array(U)    
    #z=0
    #T=10
    #sigmay=1E-4
    E=2500
    y=(np.sqrt(abs((A(U,z)**2*sigmay**2)+B(U,z)**2*(sc.k*T)/(2*sc.e*E)))) #0.506211 is conversion factor between
                                                                            #the voltage on the repel plate and the beam energy
    return y;
 
def sizefit(U, sigmay):#, yfwhm):
    U=np.array(U)    
    
    T=0
    #print(yfwhm)
    E=1132 
    z=0.49E-3
    
    y=(np.sqrt(abs((A_blur(U,z,sigmay)**2*sigmay**2)+B_blur(U,z,sigmay)**2*(sc.k*T)/(2*sc.e*E)))) #0.506211 is conversion factor between
                                                                            #the voltage on the repel plate and the beam energy
    
    return y;

'''Estimates the width and rotation of the gaussain by linear fitting the gaussain and then summing the
number of points above threshold along that line. Then that line is rotated by pi/2 and the process is repeated
for the second axis.'''


def profileanimation(outfile):
    import subprocess
    subprocess.check_call(["/usr/local/bin/ffmpeg","-framerate", "30", "-i", "img/%1d.png", 
                              "-c:v", "libx264", "-r", "30", "-pix_fmt", "yuv420p", outfile])
    subprocess.check_call(["rm", "-rf",  "img"])

def generateBG(BGpath, verbose=0):
    BG=np.genfromtxt(BGpath,skip_header=1)
    return BG

def average_shots(names,lastnames,header,lastheader):
    #print(abs(header[4]-lastheader[4]))
    if abs(header[4]-lastheader[4])<10:# and names.split()[4]==lastnames.split()[4] and:
        return True
    else:
        return False
    
def remove_folder(path):
    import os, shutil
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)
         remove_folder("/folder_name")
         
def align_shots(z,ig):
    y0=ig[1]#len(z)/2#
    x0=ig[2]#len(z[0])/2#
    zx=len(z[0])
    zy=len(z)
    xlimit=int(3*ig[3])
    ylimit=int(3*ig[3])
    #print([int(x0)-xlimit,int(x0)+xlimit,int(y0)-ylimit,int(y0)+ylimit])
    return z[np.clip(int(x0)-xlimit,0,zx):np.clip(int(x0)+xlimit,0,zx),np.clip(int(y0)-ylimit,0,zy):np.clip(int(y0)+ylimit,0,zy)]
    

def importdata(directory, verbose=0, biases=[], outdir='out', outfile='none_given',ptype='e'):
    import os
    import re
    if not os.path.exists (directory):
        print("No such folder")
        return -1
    
    if outfile == 'none_given':
        outfile = directory.split('/')[-1] 
        outdir = directory.split('/')[-3]+'/'+directory.split('/')[-2]
    
    expt_name=outdir+outfile
    bias_groups={}
    k=0
    
    try:
        for root, dirs, files in os.walk(directory):
          for names in files:
            filepath = os.path.join(root,names)
            if "header" in names:
                header=np.genfromtxt(filepath,names=True)
        
          bg_open=False
          for names in tqdm(files,smoothing=0):
            if names==".DS_Store" : continue   
            if "header" in names : continue    
        
            filepath = os.path.join(root,names)
            
            if "bg" in names:
                if bg_open == False:
                    bg=bias_group(0, 0, np.transpose(np.genfromtxt(filepath))[10:][10:])
                    bg_open=True
                    continue
                else:
                    bg.add_shot(0, 0, np.transpose(np.genfromtxt(filepath))[10:][10:])
                    continue
            
            if verbose == 1:
              print('Hashing', names)
            shot_no=int(re.split('_|\.',names)[1])
            if ptype=='i':
                bias=header["V5"][shot_no+1]#shot numbering starts from 1
            
            else:bias=header["V1"][shot_no+1]
            
            if 100<bias<5500:
                if np.around(bias,decimals=-1) in bias_groups:
                    bias_groups[np.around(bias,decimals=-1)].add_shot(header['voltage_change'][shot_no+1], bias, np.transpose(np.genfromtxt(filepath))[10:][10:])
                else:
                    bias_groups[np.around(bias,decimals=-1)]=bias_group(header['voltage_change'][shot_no+1], bias, np.transpose(np.genfromtxt(filepath))[10:][10:])
        
        
        
        k=0
        fits=[]
        for bias, group in tqdm(sorted(bias_groups.items()),smoothing=0):
            try:
                if group.shot_count<10:
                    continue
                else:
                    group.fit_gaussain(bg.x_section/bg.shot_count)
                    group.save_fig(k)
                    fits.append(group.fit)
                    k+=1
            except:
                import traceback
                # Print the stack traceback
                traceback.print_exc()
                print(bias)
                continue
            
        if os.path.exists(expt_name.split('/')[-1]+".mp4"):
            os.remove(expt_name.split('/')[-1]+".mp4")
        profileanimation(expt_name.split('/')[-1]+".mp4")
        plt.clf()
        
        #print(makefitT(biases,np.array(stdy)))
        
        #plt.savefig('out/'+expt_name+"biases_vs_stdx.svg")
        
        #Save the width and position of the fitted gaussains in terms of pixels
       
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            
        np.savetxt(outdir+'/'+outfile+'.txt', fits,header='biases stdx stdy errstdx errstdy x0 y0 theta' )
        
        return bias_groups
    except:
        import traceback
        # Print the stack traceback
        traceback.print_exc()
        print(group)
        

  
def importfolder(directory,ptype='e'):
    import os
    if not os.path.exists (directory):
        return -1
    
    for root, dirs, files in os.walk(directory):
        for folders in dirs:
            importdata(directory+'/'+folders,ptype=ptype)
            print(folders+' recorded')
            
def strip_outliers(data):
    outliers=np.where(data['errstdy']>0.02)#data['stdy']/100)
    return np.delete(data,outliers)

def strip_low_repeats(stat,count):
    outliers=np.where(count[0]<20)
    return np.delete(stat,outliers)
            
def plot_waist_scan(directory):
    import os
    import operator
    
    if not os.path.exists (directory):
        return -1
    i=0
    w=[]
    for root, dirs, files in os.walk(directory):
        for names in files:
            print(names)
            if names==".DS_Store" : continue
            if "vs" in names : continue
            wl=float(names.split('.')[0])
            if wl > 1000: wl=wl/10
            w.append(wl)
        wlrange=max(w)-min(w)
        print(wlrange)
        
        for names in files:
            if names==".DS_Store" : continue
            if "vs" in names : continue
            #extract waist scans from files
            data=np.genfromtxt(directory+'/'+names, names=True, delimiter=' ')
            #print(data)
            #extract wavelength from file name
            wavelength=float(names.split('.')[0])
            if wavelength > 1000: wavelength=wavelength/10
            
            if wlrange>0:
                cx=plt.cm.viridis((wavelength-min(w))/wlrange)
                cy=plt.cm.viridis((wavelength-min(w))/wlrange)
            else : 
                cx='C0' 
                cy='C1'
            #data=strip_outliers(data)
            
            #print(data)
            
            #plt.plot(data['biases'], data['stdx'],'x')
            
            unique, counts = np.unique(data['biases'], return_counts=True)
            
            if np.mean(counts)>2:
                grouped_mean_x=ss.binned_statistic(data['biases'], data['stdx'], statistic='mean', bins=np.unique(data['biases']))
                group_error_x=ss.binned_statistic(data['biases'], data['stdx'], statistic=np.std, bins=np.unique(data['biases']))
                group_count_x=ss.binned_statistic(data['biases'], data['stdx'], statistic='count', bins=np.unique(data['biases']))
    
                grouped_mean_y=ss.binned_statistic(data['biases'], data['stdy'], statistic='mean', bins=np.unique(data['biases']))
                group_error_y=ss.binned_statistic(data['biases'], data['stdy'], statistic=np.std, bins=np.unique(data['biases']))
                group_count_y=ss.binned_statistic(data['biases'], data['stdy'], statistic='count', bins=np.unique(data['biases']))
                
                grouped_mean_strip_x=strip_low_repeats(grouped_mean_x[0], group_count_x)
                group_error_strip_x=strip_low_repeats(group_error_x[0], group_count_x)
                biases_strip_x=strip_low_repeats(group_error_x[1][:-1], group_count_x)
                
                grouped_mean_strip_y=strip_low_repeats(grouped_mean_y[0], group_count_y)
                group_error_strip_y=strip_low_repeats(group_error_y[0], group_count_y)
                biases_strip_y=strip_low_repeats(group_error_y[1][:-1], group_count_y)
                
            else:
                n=0
                grouped_mean_strip_x=abs(data['stdx'][n:]*np.cos(data['theta'][n:]))
                group_error_strip_x=data['errstdx'][n:]
                biases_strip_x=data['biases'][n:]
                
                grouped_mean_strip_y=abs(data['stdy'][n:]*np.sin(data['theta'][n:]))
                group_error_strip_y=data['errstdy'][n:]
                biases_strip_y=data['biases'][n:]
            
            #print(group_error[0])
            #f=plt.plot(group_error_y[1][:-1], group_count_y[0], marker='.',linestyle='None',color=cx)
            
            plt.figure(3)
            
            if len(grouped_mean_strip_y)>10:      
                f=plt.errorbar(biases_strip_x, grouped_mean_strip_x/pixpermm, yerr=group_error_strip_x/pixpermm, marker='.',linestyle='None',color=cx)
                print(makefitT(biases_strip_y,grouped_mean_strip_y,color=cy,label='y')[0])
                k=plt.errorbar(biases_strip_y, grouped_mean_strip_y/pixpermm, yerr=group_error_strip_y/pixpermm, marker='.',linestyle='None',color=cy)
                print(makefitT(biases_strip_x,grouped_mean_strip_x,color=cx,label='x')[0])

            #print(min(data['stdx']/pixpermm))
            
        #handles, labels = plt.gca().get_legend_handles_labels()
        #hl = sorted(zip(handles, labels),key=operator.itemgetter(1))
        #handles2, labels2 = zip(*hl)
        #plt.gca().legend(handles2, labels2)
        
        plt.gca()
        plt.legend()
        
        plt.xlabel('Einzel lens bias (V)')
        plt.ylabel('Beam size at MCP (m)')
        #plt.ylim([0.13,0.25])
        #plt.xlim([200,600])
        
    plt.savefig('out/waist_scans.svg')
    
            

