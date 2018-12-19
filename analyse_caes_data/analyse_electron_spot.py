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
import re
import pdb
import os

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
    
def makefitT_z_sig(biases,stdy, color='b',label='None',yerr=None):
    fitT=so.curve_fit(tempfit,biases,np.array(stdy)/pixpermm,sigma=yerr,p0=[35,190E-6,0],bounds=([0,184E-6,-1.5E-5],[300,203E-6,1.5E-5]))
    plt.gcf()
#    plt.plot(biases,np.array(stdy)/pixpermm*1E6,color=color,marker='.',linestyle='None')#+ '{:.1f} $\mu$m, '.format(fitT[0][1]*1E6))
    ez=np.sqrt(np.diag(fitT[1]))[2]
    eT=np.sqrt(np.diag(fitT[1]))[0]
    esig=np.sqrt(np.diag(fitT[1]))[1]
    biases_r=np.linspace(min(biases),max(biases),50)
    fittedfunction=tempfit(biases_r,fitT[0][0],fitT[0][1],fitT[0][2])#,fitT[0][2])
    plt.plot(biases_r,fittedfunction*1E6,color=color,label=label+'nm, {:.1f} $\pm$ {:.1f} K, '.format(fitT[0][0],eT)+ '{:.1f} $\pm$ {:.1f} $\mu$m, '.format(fitT[0][1]*1E6,esig*1E6))# +'z0=''{:.1f} $\pm$ {:.1f} mm,'.format(fitT[0][2]*1E3,ez*1E3))
    plt.xlabel("Einzel lens bias / V")
    plt.ylabel("Electron bunch size / mm")
    #print(biases[np.argmin(fittedfunction)])
    #print(biases[np.argmin(stdy)])
    
    return fitT, eT, esig;

def makefitT(biases,stdy, color='b',label='None',yerr=None):
    fitT=so.curve_fit(tfit,biases,np.array(stdy)/pixpermm,sigma=yerr,p0=[20],bounds=([0],[300]))
    plt.gcf()
#    plt.plot(biases,np.array(stdy)/pixpermm*1E6,color=color,marker='.',linestyle='None')#+ '{:.1f} $\mu$m, '.format(fitT[0][1]*1E6))
#    ez=np.sqrt(np.diag(fitT[1]))[2]
    eT=np.sqrt(np.diag(fitT[1]))[0]
    esig=0#np.sqrt(np.diag(fitT[1]))[1]
    biases_r=np.linspace(min(biases),max(biases),50)
    fittedfunction=tfit(biases_r,fitT[0][0])#,fitT[0][1],fitT[0][2])#,fitT[0][2])
    plt.plot(biases_r,fittedfunction*1E6,color=color,label=label+', {:.1f} $\pm$ {:.1f} K, '.format(fitT[0][0],eT))#+ '{:.1f} $\pm$ {:.1f} $\mu$m, '.format(fitT[0][1]*1E6,esig*1E6))# +'z0=''{:.1f} $\pm$ {:.1f} mm,'.format(fitT[0][2]*1E3,ez*1E3))
    plt.xlabel("Einzel lens bias / V")
    plt.ylabel("Electron bunch size / mm")
    #print(biases[np.argmin(fittedfunction)])
    #print(biases[np.argmin(stdy)])
    
    return fitT, eT, esig;

#A_tab=np.genfromtxt('../../experimental_data/combined_DC/A_and_B/A_vs_z_vs_BE.txt')
#B_tab=np.genfromtxt('../../experimental_data/combined_DC/A_and_B/B_vs_z_vs_BE.txt')


A_tab=np.genfromtxt('../../experimental_data/28-06-18/AC/A_and_B/lens_pos/-0/A_vs_z_vs_BE.txt')
B_tab=np.genfromtxt('../../experimental_data/28-06-18/AC/A_and_B/lens_pos/-0/B_vs_z_vs_BE.txt')
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
    N=100
    x=np.linspace(z-10*sigmaz,z+10*sigmaz,N)
    g=ss.norm.pdf(x, z, sigmaz)
    g=g/max(g)
    
    for blur_z,gauss in zip(x,g):
        Ablur=A(E,blur_z)*gauss+Ablur
        
    return Ablur/sum(g)
    
def B_blur(E,z,sigmaz):
    Bblur=0
    N=100
    x=np.linspace(z-10*sigmaz,z+10*sigmaz,N)
    g=ss.norm.pdf(x, z, 2*sigmaz)
    g=g/max(g)

    for blur_z,gauss in zip(x,g):
        Bblur=B(E,blur_z)*gauss+Bblur

    return Bblur/sum(g)

def tfit(U, T):#, yfwhm):
    U=np.array(U)    
#    z=0.01E-3
    #T=0
#    sigmay=2.45E-4#1E-3
    z=0
    E=2693/2
    y=(np.sqrt(abs((A(U,z)**2*sigmay**2)+B(U,z)**2*(sc.k*T)/(2*sc.e*E)))) #0.506211 is conversion factor between
    #y=(np.sqrt(abs((A_blur(U,z,sigmaz)**2*sigmay**2)+B_blur(U,z,sigmaz)**2*(sc.k*T)/(2*sc.e*E)))) #0.506211 is conversion factor between
    return y

def tempfit(U, T, sigmay, z):#, yfwhm):
    U=np.array(U)    
#    z=0
    #T=0
#    sigmay=210E-6
#    sigmaz=sigmay#=1E-4#1E-3
    E=2693/2
    y=(np.sqrt(abs((A(U*1.091,z)**2*sigmay**2)+B(U,z)**2*(sc.k*T)/(2*sc.e*E)))) #0.506211 is conversion factor between
    #y=(np.sqrt(abs((A_blur(U,z,sigmaz)**2*sigmay**2)+B_blur(U,z,sigmaz)**2*(sc.k*T)/(2*sc.e*E)))) #0.506211 is conversion factor between
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


def profileanimation(outfile,imgfolder):
    import subprocess
    subprocess.check_call(["/usr/local/bin/ffmpeg","-framerate", "30", "-i", imgfolder+"/%1d.png", 
                              "-c:v", "libx264", "-r", "30", "-pix_fmt", "yuv420p", outfile])
    subprocess.check_call(["rm", "-rf",  imgfolder])

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
         
#def align_shots(z,ig):
#    y0=ig[1]#len(z)/2#
#    x0=ig[2]#len(z[0])/2#
#    zx=len(z[0])
#    zy=len(z)
#    xlimit=int(3*ig[3])
#    ylimit=int(3*ig[3])
#    #print([int(x0)-xlimit,int(x0)+xlimit,int(y0)-ylimit,int(y0)+ylimit])
#    return z[np.clip(int(x0)-xlimit,0,zx):np.clip(int(x0)+xlimit,0,zx),np.clip(int(y0)-ylimit,0,zy):np.clip(int(y0)+ylimit,0,zy)]
#    

def importdata(directory, verbose=0, biases=[], outdir='out', outfile='none_given',ptype='e',decimation=1):
    if not os.path.exists (directory):
        print("No such folder")
        return -1
    
        
    bias_groups={}
    k=0
    bg_open=False
    try:
        for root, dirs, files in os.walk(directory+'/../'):
            for names in files:
                if "bg" in names:
                    filepath = os.path.join(root,names)
                    if bg_open == False:
                        bg=bias_group(0, 0, np.transpose(np.genfromtxt(filepath))[5:,5:])#[10:][10:])
                        bg_open=True
                        continue
                    else:
                        bg.add_shot(0, 0, np.transpose(np.genfromtxt(filepath))[5:,5:])#[10:][10:])
                        continue
    except:
        print(root)
        import traceback
        # Print the stack traceback
        traceback.print_exc()

    
    try:
        for root, dirs, files in os.walk(directory):
            for names in files:
                filepath = os.path.join(root,names)
                if "header" in names:
                    header=np.genfromtxt(filepath,names=True)
    

            for names in tqdm(files[0::decimation],smoothing=0):
                if names==".DS_Store" : continue   
                if "header" in names : continue    
                if "bg" in names : continue
                if "profile.mp4" in names: continue
            
                filepath = os.path.join(root,names)
                
                if verbose == 1:
                  print('Hashing', names)
                shot_no=int(re.split('_|\.',names)[-2])
                
                if shot_no>len(header["V5"]): continue
                
                if ptype=='i':
                    bias=header["V5"][shot_no-1]#shot numbering starts from 1
                
                else:bias=header["V4"][shot_no-1]
                rounding=0
                group=shot_no
                if 100<bias<5000 and header['voltage_change'][shot_no-1]==0:
                    if np.around(group,decimals=rounding) in bias_groups:
                        bias_groups[np.around(group,decimals=rounding)].add_shot(header['voltage_change'][shot_no-1], bias, np.transpose(np.genfromtxt(filepath))[5:,5:])#,z0=header['z0'][shot_no-1],errz0=header['errz0'][shot_no-1])#,modelstdx=header['stdx'][shot_no-1])#[10:][10:])
                    else:
                        bias_groups[np.around(group,decimals=rounding)]=bias_group(header['voltage_change'][shot_no-1], bias, np.transpose(np.genfromtxt(filepath))[5:,5:])#,z0=header['z0'][shot_no-1],errz0=header['errz0'][shot_no-1])#,modelstdx=header['stdx'][shot_no-1])#[10:][10:])
        
        return bias_groups, bg, bg_open
    
    except:
        import traceback
        # Print the stack traceback
        traceback.print_exc()
        print(group)

def fit_run(bias_groups, bg, bg_open, directory="out/test/temp",outfile='none_given'):
    
    outdir = directory+'/../results'
    outdir_1d = directory+'/../results_1d'
    viddir = directory+'/../vids'
    
    if outfile == 'none_given':
        outfile = directory.split('/')[-1] 
    
    expt_name=outdir+outfile
    
    if not os.path.exists(viddir):
        os.makedirs(viddir)
        
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    if not os.path.exists(outdir_1d):
        os.makedirs(outdir_1d)
        
    try:
        k=0
        fits=[]
        fits1d=[]
        if bg_open==True:
            bg_norm=bg.x_section/200 #200 shots in background image
        else:
            bg_norm=0
            
        for bias, group in tqdm(sorted(bias_groups.items()),smoothing=0):
            try:
                if group.shot_count<1:
                    continue
                if group.shot_count>1:
                    group.align_shots(bg_norm)
                if group.shot_count==1:
                    group.single_shot(bg_norm)

                group.fit_gaussain(bg_norm)
#                    group.fit_1d_gaussain(bg_norm)
                group.save_fig(k)
                group.transverse(k)
                fits.append(group.fit)
#                    fits1d.append(group.fit1d)
                k+=1
            except:
                import traceback
                # Print the stack traceback
                traceback.print_exc()
                print(bias)
                #group.initial_guess(bg_norm)
                group.transverse(k)
                group.save_fig(k)
                k+=1
                plt.clf()
                continue
    
        np.savetxt(outdir+'/'+outfile+'.txt', fits, header='biases shot_count stdx stdy errstdx errstdy x0 y0 theta modelstdx e_count z0 errz0' )
        np.savetxt(outdir_1d+'/'+outfile+'.txt', fits1d, header='biases shot_count stdx stdy errstdx errstdy x0 y0 theta modelstdx e_count z0 errz0' )
        
        if os.path.exists(viddir+'/'+outfile+".mp4"):
            os.remove(viddir+'/'+outfile+".mp4")
        profileanimation(viddir+'/'+outfile+".mp4",'img')
        
        if os.path.exists(viddir+'/t'+outfile+".mp4"):
            os.remove(viddir+'/t'+outfile+".mp4")
        profileanimation(viddir+'/t'+outfile+".mp4",'imgt')
        plt.clf()
            
            #print(makefitT(biases,np.array(stdy)))
            
            #plt.savefig('out/'+expt_name+"biases_vs_stdx.svg")
            
            #Save the width and position of the fitted gaussains in terms of pixels
                
            
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
            if 'results' in folders:
                continue
            if 'vids' in folders:
                print(folders)
                continue
            if 'bg' in folders:
                print(folders)
                continue
            if 'A_and_B' in folders:
                print(folders)
                continue
            if 'TOF' in folders:
                print(folders)
                continue
            else:
                fit_run(*importdata(directory+'/'+folders,ptype=ptype,decimation=1),directory=directory+'/'+folders)
                print(folders+' recorded')
            
def strip_outliers(data):
    outliers=np.where((data['errstdy']>1000) | (data['stdy']>1000) | (data['biases']>4800))#data['stdy']/100)
    return np.delete(data,outliers)

def strip_low_repeats(stat,count):
    outliers=np.where(count[0]<50)
    return np.delete(stat,outliers)

def ellipse(a,b,theta,theta_eval):
    return a*b/(np.sqrt(b*np.cos(theta_eval-theta)**2+a*np.sin(theta_eval-theta)**2))
    
def plot_waist_scan(directory):
    import os
    
    if not os.path.exists (directory):
        return -1
    
    w=[]
    e_c=[]
    F=[]
    MOT_sig_y=[]
    bias=[]
    means=[]
    error=[]
    for root, dirs, files in os.walk(directory):
        for names in files:
            print(names)
            if names==".DS_Store" : continue
            if "vs" in names : continue
            wl=float(re.split('_|\.',names)[0])+float(re.split('_|\.',names)[1])/100
            w.append(wl)
            data=np.genfromtxt(directory+'/'+names, names=True, delimiter=' ')
            e_c.append(np.mean(data['e_count']/4.06))
            
        wlrange=max(w)-min(w)
        e_crange=max(e_c)-min(e_c)
        print(wlrange)
        T=[]
        Terr=[]
        for names in files:
            if names==".DS_Store" : continue
            if "vs" in names : continue
            #extract waist scans from files
            data=np.genfromtxt(directory+'/'+names, names=True, delimiter=' ')
            #print(data)
            #extract wavelength from file name
            wavelength=float(re.split('_|\.',names)[0])+float(re.split('_|\.',names)[1])/100
            e_count=np.mean(data['e_count'])
            
            if abs(wlrange)>0:
                cx=plt.cm.viridis((wavelength-min(w))/wlrange)
                cy=plt.cm.viridis((wavelength-min(w))/wlrange)
            else : 
                cx='C0'#plt.cm.viridis((e_count-min(e_c))/e_crange)
                cy='C1'#plt.cm.viridis((e_count-min(e_c))/e_crange)
            #data=strip_outliers(data)
            
            #print(data)
            
#            plt.plot(data['biases'], data['stdx'],'x')
            
            unique, counts = np.unique(data['biases'], return_counts=True)
            
#            print('ec={:}'.format(e_count))
            
            data=strip_outliers(data)
            
            bias_round=-1
            
            if np.mean(counts)>1:
                grouped_mean_x=ss.binned_statistic(np.around(data['biases'],decimals=bias_round), data['stdx'], statistic='mean', bins=np.unique(np.around(data['biases'],decimals=bias_round)))
                group_error_x=ss.binned_statistic(np.around(data['biases'],decimals=bias_round), data['stdx'], statistic=np.std, bins=np.unique(np.around(data['biases'],decimals=bias_round)))
                group_count_x=ss.binned_statistic(np.around(data['biases'],decimals=bias_round), data['stdx'], statistic='count', bins=np.unique(np.around(data['biases'],decimals=bias_round)))
    
                grouped_mean_y=ss.binned_statistic(np.around(data['biases'],decimals=bias_round), data['stdy'], statistic='mean', bins=np.unique(np.around(data['biases'],decimals=bias_round)))
                group_error_y=ss.binned_statistic(np.around(data['biases'],decimals=bias_round), data['stdy'], statistic=np.std, bins=np.unique(np.around(data['biases'],decimals=bias_round)))
                group_count_y=ss.binned_statistic(np.around(data['biases'],decimals=bias_round), data['stdy'], statistic='count', bins=np.unique(np.around(data['biases'],decimals=bias_round)))
                
                grouped_mean_strip_x=strip_low_repeats(grouped_mean_x[0], group_count_x)
                group_error_strip_x=strip_low_repeats(group_error_x[0], group_count_x)
                biases_strip_x=strip_low_repeats(group_error_x[1][:-1], group_count_x)
                
                grouped_mean_strip_y=strip_low_repeats(grouped_mean_y[0], group_count_y)
                group_error_strip_y=strip_low_repeats(group_error_y[0], group_count_y)
                biases_strip_y=strip_low_repeats(group_error_y[1][:-1], group_count_y)
                
            else:
                n=0
                grouped_mean_strip_x=data['stdx'][n:]#ellipse(np.array(data['stdx'][n:]),np.array(data['stdy'][n:]),np.array(data['theta'][n:]),0)
                group_error_strip_x=data['errstdx'][n:]
                biases_strip_x=data['biases'][n:]
            
                grouped_mean_strip_y=data['stdy'][n:]#ellipse(np.array(data['stdx'][n:]),np.array(data['stdy'][n:]),np.array(data['theta'][n:]),np.pi/2)
                group_error_strip_y=data['errstdy'][n:]
                biases_strip_y=data['biases'][n:]
            
            #print(group_error[0])
            #f=plt.plot(group_error_y[1][:-1], group_count_y[0], marker='.',linestyle='None',color=cx)
            
            plt.figure(3)
            n=0
            m=10000#len(grouped_mean_strip_y)
            #print(len(grouped_mean_strip_y))

            if len(grouped_mean_strip_y)>1: 
#                plt.errorbar(biases_strip_y[n:m], grouped_mean_strip_y[n:m]/pixpermm*1E6, yerr=group_error_strip_y[n:m]/pixpermm*1E6, marker='.',linestyle='None',color=cx)#,label='electron count = {:.3g} $\pm$ {:.3g}'.format(np.mean(data['e_count']),np.std(data['e_count'])))
#                T_fit,T_fiterr,sigerr=makefitT_z_sig(biases_strip_y[n:m],grouped_mean_strip_y[n:m],color=cx,label='{:.4g}'.format(wavelength),yerr=group_error_strip_y[n:m]/pixpermm)#np.mean(data['e_count']))))
#                T_fit=F[-1]
                plt.errorbar(biases_strip_x[n:m], grouped_mean_strip_x[n:m]/pixpermm*1E6, yerr=group_error_strip_x[n:m]/pixpermm*1E6, marker='.',linestyle='None',color=cx)#,label='electron count = {:.3g} $\pm$ {:.3g}'.format(np.mean(data['e_count']),np.std(data['e_count'])))
                T_fit,T_fiterr,sigerr=makefitT_z_sig(biases_strip_x[n:m],grouped_mean_strip_x[n:m],color=cx,label='{:.4g}'.format(wavelength),yerr=group_error_strip_x[n:m]/pixpermm)#np.mean(data['e_count']))))

                F.append(T_fit)
                print('wl={}, T={}, sig={}, z0={}'.format(wavelength,*T_fit[0]))
                T.append(T_fit[0])
                Terr.append([T_fiterr,sigerr])
#                bias.append(biases_strip_x[n:m])
#                means.append(grouped_mean_strip_x[n:m]/pixpermm)
#                error.append(group_error_strip_x[n:m]/pixpermm)
                
#               MOT_sig_y.append(np.mean(data['sigy']))
    
        #print(min(data['stdx']/pixpermm))
#        plt.ylim(0,500)
#        print(F)
#        plt.gca()
        handles, labels = plt.gca().get_legend_handles_labels()
        
        # or sort them by labels

        hl = sorted(zip(handles, labels, e_c),
                    key=operator.itemgetter(1))
        handles2, labels2, e_c2 = zip(*hl)
#        lgd=plt.legend(handles, labels,loc="upper left", bbox_to_anchor=(1,1))
        lgd=plt.legend(handles2, labels2,loc="upper left", bbox_to_anchor=(1,1))
#        plt.tight_layout()
        
#        pdb.set_trace()
        plt.xlabel('Einzel lens bias (V)')
        plt.ylabel('Beam size at MCP ($\mu$m)')
        plt.text(5500,100,"z position = {:.3f} $\pm$ {:.3f} $\mu$m \nsource size = {:.1f} $\pm$ {:.1f} $\mu$m \nTemp error ave = {:.1f} K".format(np.mean(np.array(T)[:,2])*1E6,np.std(np.array(T)[:,2])*1E6,np.mean(np.array(T)[:,1])*1E6,np.std(np.array(T)[:,1])*1E6,np.mean(np.array(Terr)[:,0])))
#        plt.ylim([0.13,0.25])
#        plt.xlim([200,600])
        
    plt.savefig(directory+'waist_scans.pdf', dpi=300,bbox_extra_artists=(lgd,), bbox_inches='tight')
    
    return T#[bias, means, error]
            

