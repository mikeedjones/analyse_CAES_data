"""
Created on Sat Jul 22 14:15:22 2017

@author: MichTOFael
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import mode
import scipy.interpolate as si
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

dt0=2.000000E-10

BE_table=np.zeros([10,10])

TOF_table=np.zeros([10,10])

Ext=np.array(TOF_table[1:-1,0])

zarr=np.array(TOF_table[0,1:])*1e3

time=np.array(TOF_table[1:-1,1:])*1e9

Ext=np.array(BE_table[1:-1,0])

zarr=np.array(BE_table[0,1:])*1e3

energy=np.array(BE_table[1:-1,1:])

def two_scales(ax1, time, data1, data2, c1, c2):
    """

    Parameters
    ----------
    ax : axis
        Axis to put two scales on

    time : array-like
        x-axis values for both datasets

    data1: array-like
        Data for left hand scale

    data2 : array-like
        Data for right hand scale

    c1 : color
        Color for line 1

    c2 : color
        Color for line 2

    Returns
    -------
    ax : axis
        Original axis
    ax2 : axis
        New twin axis
    """
    ax2 = ax1.twinx()

    ax1.plot(time, data1, color=c1)
    ax2.plot(time, data2, color=c2)

    return ax1, ax2
#checks whether the data has been clipped
def check_threshold(y):
    A,counts=np.unique(y, return_counts=True)
    if counts[-1]>10:
#        plt.plot(y)
#        plt.title(len(np.where(y==max(y))[0]))
#        plt.show()
        return True
    else:
        return False

# Define model function to be used to fit to the data above:
def gauss(x, A, mu, sigma, o):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))+o

# p0 is the initial guess for the fitting coefficients (A, mu and sigma)
def guessgauss(y,dt):
    A=max(y)
    mu=np.argmax(y)
    o=np.median(y)
    
    i=mu
    
    while y[i]>o:
        i+=1

    sigma=(i-mu)/2.35
    
    return [A,mu*dt,sigma*dt, o]

#strip the data of anything except the initial peak
def clean_data(y, p0, dt):
    try:
        n=len(y)-int(p0[1]/dt)-2*int(p0[2]/dt)-1
        y=np.append(y[0:len(y)-n],[p0[3]]*n)
        y=np.clip(y,0,3*p0[0])
    except:
        print(p0, dt)
    return y

def clean_fit_gaussain_data(y,dt):
    t=np.arange(0,dt*len(y),dt)
    
    p0=guessgauss(y,dt)
    
#    if p0[2]==0:
#        plt.plot(y)
#        plt.title(len(np.where(y==max(y))[0]))
##        plt.ylim((max(y)*0.99,max(y)*1.01))
#        plt.show()

    y=clean_data(y,p0, dt)
    
    fit=curve_fit(gauss, t, y, p0=p0)
    
#    plt.plot(t*1e9, gauss(t,*fit[0]), 'C0')
#    plt.plot(t*1e9, y, 'C1')
#    plt.show()
#    plt.clf()
    
    
    return fit;

def TOFcurve(E,z,deltat):
    return si.RectBivariateSpline(Ext,zarr,time).ev(E,z)+deltat

def BEcurve(E,z):
    return si.RectBivariateSpline(Ext,zarr,energy).ev(E,z)
    
def fitTOF(extV,t):
    return curve_fit(TOFcurve,extV,t,bounds=([0,-4],[1,-3]),p0=[0.3,-3.5]);

def load_TOF_model(TOF_mod_dir):
    global TOF_table 
    TOF_table = np.genfromtxt(TOF_mod_dir+'/z_v_Ext_v_TOF.txt')
    global BE_table
    BE_table = np.genfromtxt(TOF_mod_dir+'/z_v_Ext_v_BE.txt')
    global Ext
    Ext=np.array(TOF_table[1:,0])
    global zarr
    zarr=np.array(TOF_table[0,1:])*1e3
    global time
    time=np.array(TOF_table[1:,1:])*1e9
    global energy
    energy=np.array(BE_table[1:,1:])

def show_TOF_model(TOF_mod_dir):
    plt.imshow(time, aspect='auto')
    plt.xticks(np.linspace(0,len(zarr),5),np.linspace(min(zarr),max(zarr),5))
    plt.yticks(np.linspace(0,len(Ext),5),np.linspace(min(Ext),max(Ext),5))
    plt.xlabel('Starting z position (mm)')
    plt.ylabel('Extraction volatge (V)')
    plt.colorbar(label='TOF (ns)')

def show_fits(t,e,b,efit,bfit):
    fig, ax = plt.subplots()
    ax1, ax2 = two_scales(ax, t*1e9, b, e, 'C0', 'C1')
    
    ax1.set_xlabel('time (ns)')
    ax1.set_ylabel('Blue laser pulse (a.u.)')

    ax1.plot(t*1e9, gauss(t,*bfit[0]), 'C0')
    ax1.plot(t*1e9, b, 'C0')

    ax2.set_ylabel('Electron pulse (a.u.)')
    ax2.plot(t*1e9, gauss(t,*efit[0]), 'C1')
    ax2.plot(t*1e9, e, 'C1')
    
#    plt.title('Extraction bias = {} V.svg'.format(float(name[:-4])))
    #plt.savefig('scope_traces/{} V.svg'.format(float(name[:-4])))
    #plt.close()
    plt.show()

def TOF_versus_extV(directory,savefig='TOF v extraction voltage.svg',disable_prog_bar=False,verbose=0,TOF_mod_dir='../2k'):
    
    load_TOF_model(TOF_mod_dir)
    
    TOF_full=[]
    V_full=[]
    
    TOF=[]
    V=[]
    eTOF=[]
    dt=0
    lengths=[]
    
    if not os.path.exists (directory):
        return -1
    
    try:
        for root, dirs, files in os.walk(directory):
          for name in tqdm(files,smoothing=0,disable=disable_prog_bar):
            if name==".DS_Store" : continue   
            if verbose == 1:
                print('Hashing', name)
            filepath = os.path.join(root,name)
            
            b=[]
            e=[]
            TOFtemp=[]
            flag=False
            
            with open(filepath) as f:
                for line in f:
                    if 'delta' in line and dt==0:
                        dt=float(line.split()[2])
                        flag=False
                    if 'waveform' in line and e!=[]:
                        flag=False
                        if check_threshold(e):
                            e=[]
                            b=[]
                            continue
                        
                        fitElectron=(clean_fit_gaussain_data(e,dt))
                        fitBlue=(clean_fit_gaussain_data(b,dt))
                        
                        t=np.arange(0,dt*len(e),dt)
                        
                        e=[]
                        b=[]
                        TOFtemp.append(fitElectron[0][1]-fitBlue[0][1])
    

                    if flag==True:
                        b.append(float(line.split()[2]))
                        e.append(float(line.split()[3]))
                    if 'time' in line:
                        flag=True

            if np.std(e)<0.001 or np.std(b)<0.002:
                e=[]
                b=[]

                continue
            fitElectron=(clean_fit_gaussain_data(e,dt))
            fitBlue=(clean_fit_gaussain_data(b,dt))
            
            e=[]
            b=[]            

            TOFtemp.append(fitElectron[0][1]-fitBlue[0][1])
            if len(TOFtemp)>3:
                for l in TOFtemp:
                    TOF_full.append(l)
                    V_full.append(float(name[:-4]))
                V.append(float(name[:-4]))
                TOF.append(np.mean(TOFtemp))
                lengths.append(len(TOFtemp))
                eTOF.append(np.std(TOFtemp))#/np.sqrt(len(TOFtemp)))
            #y=np.loadtxt(filepath, skiprows=5, usecols=(2,3))

    except:
        print(name)
        import traceback
        # Print the stack traceback
        traceback.print_exc()
        
        fig, ax = plt.subplots()
        ax1, ax2 = two_scales(ax, t*1e9, b, e, 'C0', 'C1')
        
        ax1.set_xlabel('time (ns)')
        ax1.set_ylabel('Blue laser pulse (a.u.)')

        ax2.set_ylabel('Electron pulse (a.u.)')
        
        plt.title('Extraction bias = {} V.svg'.format(float(name[:-4])))
        #plt.savefig('scope_traces/{} V.svg'.format(float(name[:-4])))
        #plt.close()
        plt.show()
    
    #scales the Times of flight and fits for delta t and z

    plt.errorbar(V,np.array(TOF)*1e9,yerr=np.array(eTOF)*1e9,marker='.',linestyle='None')    
    
    TOF_final_fit=fitTOF(V_full,np.array(TOF_full)*1e9)
    z=TOF_final_fit[0][0]
    deltat=TOF_final_fit[0][1]
    
    #calcualtes the error on the fit as the diagonal of the covariance matrix
    zerror=np.sqrt(np.diag(TOF_final_fit[1]))[0]
    deltaterror=np.sqrt(np.diag(TOF_final_fit[1]))[1]
    
    V_fit=np.linspace(min(V),max(V),1000)
    BE=BEcurve(np.array(V),TOF_final_fit[0][0])
    BE_fit=BEcurve(V_fit,TOF_final_fit[0][0])
    
    #plt.plot(V_fit,BE_fit)
    #plt.show()
    
    BEperV=np.mean(BE/V)
    BEperVerr=0.1
    
    zrange=np.linspace(-5,5,5)
    
    plt.errorbar(V,np.array(TOF)*1e9,yerr=np.array(eTOF)*1e9,marker='.',linestyle='None')    
    plt.plot(V_fit,TOFcurve(V_fit,TOF_final_fit[0][0],TOF_final_fit[0][1]),label='k={:.3} $\pm$ {:.2}, $\Delta$t={:.3} $\pm$ {:.2} ns'.format(TOF_final_fit[0][0],zerror,TOF_final_fit[0][1],deltaterror))
#    plt.plot(V_fit,TOFcurve(V_fit,0.3,TOF_final_fit[0][1]),label='k=0.3')
#    plt.plot(V_fit,TOFcurve(V_fit,-0.5,TOF_final_fit[0][1]),label='k=1.2')
    #for z in zrange:
        #plt.plot(V_fit,TOFcurve(V_fit,TOF_final_fit[0][0],z),label='$\Delta$t={} ns'.format(z))
    
    plt.ylabel('Time Difference (ns)')
    plt.xlabel('Extraction electrode voltage (V)')
    #plt.text(1200,55,'Beam energy per volt on extraction \n electode = {:.3} $\pm$ {:.2} eV/V'.format(,BEperVerr),horizontalalignment='right')
    plt.legend()
    plt.savefig(TOF_mod_dir+'/'+savefig)
        
    return TOF_final_fit;
