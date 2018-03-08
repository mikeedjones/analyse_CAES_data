# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 14:15:22 2017

@author: Michael
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import mode
import scipy.interpolate as si
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

TOF_table=np.genfromtxt('07-03-18/z_v_Ext_v_TOF.txt')
BE_table=np.genfromtxt('07-03-18/z_v_Ext_v_BE.txt')

dt0=2.000000E-10

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

# Define model function to be used to fit to the data above:
def gauss(x, A, mu, sigma, o):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))+o

# p0 is the initial guess for the fitting coefficients (A, mu and sigma)
def guessgauss(y,dt):
    A=max(y)
    mu=np.argmax(y)
    o=mode(y)[0]
    
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
    
    if p0[2]==0:
        plt.plot(t,y)
        plt.show()

    y=clean_data(y,p0, dt)
    
    fit=curve_fit(gauss, t, y, p0=p0)
    
#    plt.plot(t*1e9, gauss(t,*fit[0]), 'C0')
#    plt.plot(t*1e9, y, 'C1')
#    plt.show()
#    plt.clf()
    
    
    return fit;

Ext=np.array(TOF_table[1:-1,0])

zarr=np.array(TOF_table[0,1:])*1e3

time=np.array(TOF_table[1:-1,1:])*1e9

def TOFcurve(E,z,deltat):
    return si.RectBivariateSpline(Ext,zarr,time).ev(E,z)+deltat

Ext=np.array(BE_table[1:-1,0])

zarr=np.array(BE_table[0,1:])*1e3

energy=np.array(BE_table[1:-1,1:])

def BEcurve(E,z):
    return si.RectBivariateSpline(Ext,zarr,energy).ev(E,z)
    
def fitTOF(extV,t):
    return curve_fit(TOFcurve,extV,t,bounds=(-30,30));

def TOF_versus_extV(directory,savefig='TOF v extraction voltage.svg',verbose=0):
    
    TOF=[]
    V=[]
    eTOF=[]
    dt=0
    lengths=[]
    
    if not os.path.exists (directory):
        return -1
    
    try:
        for root, dirs, files in os.walk(directory):
          for name in tqdm(files,smoothing=0):
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
                        if np.std(e)<0.1:
                            e=[]
                            b=[]
                            continue
                        
                        fitElectron=(clean_fit_gaussain_data(e,dt))
                        fitBlue=(clean_fit_gaussain_data(b,dt))
                        
                        t=np.arange(0,dt*len(e),dt)
                        
                        e=[]
                        b=[]
                        TOFtemp.append(fitElectron[0][1]-fitBlue[0][1]+1E-9)
    

                    if flag==True:
                        b.append(float(line.split()[2]))
                        e.append(float(line.split()[3]))
                    if 'time' in line:
                        flag=True

            if np.std(e)<0.0010:
                e=[]
                b=[]

                continue
            fitElectron=(clean_fit_gaussain_data(e,dt))
            fitBlue=(clean_fit_gaussain_data(b,dt))
            
            e=[]
            b=[]            
            
            V.append(float(name[:-4]))
            TOFtemp.append(fitElectron[0][1]-fitBlue[0][1]+1E-9)
            TOF.append(np.mean(TOFtemp))
            lengths.append(len(TOFtemp))
            eTOF.append(np.std(TOFtemp)/np.sqrt(len(TOFtemp)))
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

    TOF_final_fit=fitTOF(V,np.array(TOF)*1e9)
    z=TOF_final_fit[0][0]
    deltat=TOF_final_fit[0][1]
    
    #calcualtes the error on the fit as the diagonal of the covariance matrix
    zerror=np.sqrt(np.diag(TOF_final_fit[1]))[0]
    deltaterror=np.sqrt(np.diag(TOF_final_fit[1]))[0]
    
    V_fit=np.linspace(min(V),max(V),1000)
    BE=BEcurve(V,TOF_final_fit[0][0])
    BE_fit=BEcurve(V_fit,TOF_final_fit[0][0])
    
    #plt.plot(V_fit,BE_fit)
    #plt.show()
    
    BEperV=np.mean(BE/V)
    BEperVerr=0.1
    
    zrange=np.linspace(-5,5,5)
    
    plt.clf()
    plt.errorbar(V,np.array(TOF)*1e9,yerr=np.array(eTOF)*1e9,marker='.',linestyle='None')    
    plt.plot(V_fit,TOFcurve(V_fit,TOF_final_fit[0][0],TOF_final_fit[0][1]),label='z={:.3} $\pm$ {:.2} mm, $\Delta$t={:.3} $\pm$ {:.2} ns'.format(TOF_final_fit[0][0],zerror,TOF_final_fit[0][1],deltaterror))
    #for z in zrange:
        #plt.plot(V_fit,TOFcurve(V_fit,TOF_final_fit[0][0],z),label='$\Delta$t={} ns'.format(z))
    
    plt.ylabel('Time Difference (ns)')
    plt.xlabel('Extraction electrode voltage (V)')
    plt.text(4000,50,'Beam energy per volt on extraction \n electode = {:.3} $\pm$ {:.2} eV/V'.format(BEperV,BEperVerr),horizontalalignment='right')
    plt.legend()
    plt.savefig(savefig)
        
    return TOF_final_fit;
