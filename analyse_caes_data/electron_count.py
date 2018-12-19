# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 13:02:15 2018

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
import scipy.ndimage.measurements as scim
import os

def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def gauss_functions(x1, a1, x01, sigma1, a2, x02, sigma2, a3, x03, sigma3, a4, x04, sigma4):
    return a1*np.exp(-(x1-x01)**2/(2*sigma1**2))+a2*np.exp(-(x1-x02)**2/(2*sigma2**2))+a3*np.exp(-(x1-x03)**2/(2*sigma3**2))+a4*np.exp(-(x1-x04)**2/(2*sigma4**2))
            
def count_electrons(indir):
    e_count=[]
    shotno=[]
    bg=np.transpose(np.genfromtxt('bg/bg.asc'))[10:][10:]/200
    for root, dirs, files in os.walk(indir):
        for names in tqdm(files,smoothing=0):
            if names==".DS_Store" : continue               
            if "header" in names : continue   
            filepath = os.path.join(root,names)
            shotno.append(names.split('_|.')[2])
            shot=np.clip(np.transpose(np.genfromtxt(filepath))[10:][10:]-bg,0,1E10)
            e_count.append(sum(sum(shot))/93.9)
            #plt.imshow(shot)
            #plt.colorbar()
            #plt.show()
            #plt.clf()
    #print("count = {:.1f} $\pm$ {:.1f}".format(np.mean(e_count),np.std(e_count)))
    
    np.savetxt('shot_summed_cam.txt',np.array([shotno,e_count]).T)
    
    return e_count

def electron_count(indir):
    dist=[]
    bg=np.transpose(np.genfromtxt(indir+'/../bg/bg.asc'))[10:][10:]/200
    for root, dirs, files in os.walk(indir):
        for names in tqdm(files,smoothing=0):
            if names==".DS_Store" : continue               
            if "header" in names : continue   
            filepath = os.path.join(root,names)
            shot=np.clip(np.transpose(np.genfromtxt(filepath))[10:][10:]-bg,0,1E10)
            amplitude=max(shot.flatten())
            offset=np.percentile(shot.flatten(),5)
            threshold_value=(amplitude-offset)*0.3+offset
            
            labled_array=scim.label(np.clip(shot,threshold_value,amplitude)-threshold_value)[0]
            
            for i in range(1,len(labled_array)):
#                    plt.scatter(np.where(labled_array==i)[1],np.where(labled_array==i)[0])
                if len(np.where(labled_array==i)[0])>0:
                    dist.append(sum(shot[np.where(labled_array==i)[0],np.where(labled_array==i)[1]]))
                    #plt.imshow(shot[min(np.where(labled_array==i)[0]-10):max(np.where(labled_array==i)[0])+10,min(np.where(labled_array==i)[1])-10:max(np.where(labled_array==i)[1])+10])

                    #plt.show()
                    #plt.clf() 
#                plt.imshow(shot)
    h=plt.hist(dist,bins=100,range=(0,1500))
    fit_1e(h)
    
    return dist
                
def fit_1e(dist):
    a1=max(dist[0])
    x01=dist[1][np.argmax(dist[0])]
    sigma1=x01
    a2=max(dist[0])/2
    x02=2*dist[1][np.argmax(dist[0])]
    sigma2=x01
    a3=max(dist[0])/3
    x03=3*dist[1][np.argmax(dist[0])]
    sigma3=x01*2
    a4=max(dist[0])/3
    x04=3*dist[1][np.argmax(dist[0])]
    sigma4=x01*3
    
    bin_centers=dist[1][1:]-dist[1][0]
    
    p=so.curve_fit(gauss_functions,bin_centers,dist[0],bounds=(0,1E4),p0=[a1, x01, sigma1,a2, x02, sigma2, a3, x03, sigma3,a4, x04, sigma4]);            
    #plt.plot(dist[1][1:],dist[0])
    
    print(p[0])
    
    plt.plot(bin_centers,gauss_functions(dist[1][1:],*p[0]))
    plt.xlabel('Intensity of spot/arb')
    plt.ylabel('counts')
    plt.text(600,540,'Intensity of 1 electron = {:.1f} $\pm$ {:.1f}'.format(p[0][1],p[0][2]))
    plt.text(600,500,'Intensity of 2 electrons = {:.1f} $\pm$ {:.1f}'.format(p[0][4],p[0][5]))
    plt.savefig('single_electron_intesnity.svg',dpi=300)
                
                
                