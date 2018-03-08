# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 14:07:27 2018

bias class to contain all the data from the shots taken at that bias and perform calculations on them

@author: Michael
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
import scipy.ndimage.measurements as scim
import scipy.stats as scis
import scipy.constants as sc
import operator
import os

from scipy.ndimage.filters import gaussian_filter

'''Estimates the width and rotation of the gaussain by linear fitting the gaussain and then summing the
number of points above threshold along that line. Then that line is rotated by pi/2 and the process is repeated
for the second axis.'''


def find_axis(xyz,xo,yo,threshold_value):
    xa=np.array([-1000,1000])
    m_arr=[]
    l_arr=[]
    #plt.imshow(xyz)
    #plt.scatter(boundingbox[1],boundingbox[0],color='C3')
    
    for theta in np.arange(-sc.pi,sc.pi,0.1):
        ya=np.tan(theta)*(xa-xo)+yo
        length = int(np.hypot(xa[1]-xa[0], ya[1]-ya[0]))
        if length < 100000:
            xt, yt = np.linspace(xa[0], xa[1], length), np.linspace(ya[0], ya[1], length)
            delete=np.array([*np.where(xt<0)[0], *np.where(xt>=len(xyz[0]))[0], *np.where(yt<0)[0], *np.where(yt>=len(xyz)-2)[0]])
            yt=np.delete(yt,delete.astype(np.int))
            xt=np.delete(xt,delete.astype(np.int))
            l_t = sum(np.where(xyz[yt.astype(np.int), xt.astype(np.int)]> threshold_value, 1, 0))
            l_arr.append(l_t)
            m_arr.append(np.tan(theta))
            #plt.plot(xt,yt)
            
    index, l_major = max(enumerate(l_arr), key=operator.itemgetter(1))
    
    m_at_max_l=m_arr[index]
    
    #calculate the width of the minor axis
    
    yr=-1/m_at_max_l*(xa-xo)+yo
    length = int(np.hypot(xa[1]-xa[0], yr[1]-yr[0]))
    if length < 100000:
        xt, yt = np.linspace(xa[0], xa[1], length), np.linspace(yr[0], yr[1], length)
        delete=np.array([*np.where(xt<0)[0], *np.where(xt>=len(xyz[0]))[0], *np.where(yt<0)[0], *np.where(yt>=len(xyz))[0]])
        yt=np.delete(yt,delete.astype(np.int))
        xt=np.delete(xt,delete.astype(np.int))
    
    l_minor=sum(np.where(xyz[yt.astype(np.int), xt.astype(np.int)]> threshold_value, 1, 0))
    #plt.plot(xt,yt)
    #plt.show()
    
    return l_major, l_minor, sc.pi-np.arctan(m_at_max_l)
    

def initial_guess(xyz,threshold_perc=95,flag=False):
    xyz_blurred=gaussian_filter(xyz, sigma=len(xyz[0])/25)
    amplitude=max(xyz_blurred[10:,10:].flatten())
    offset=np.percentile(xyz_blurred.flatten(),5)
    #threshold_value=np.percentile(xyz_blurred.flatten(),5)
    #if flag==True:
    threshold_value=(amplitude-offset)*0.80+offset
    
    labled_array=scim.label(np.clip(xyz_blurred,threshold_value,amplitude)-threshold_value)[0]
#    if flag==True:
#        plt.scatter(np.where(labled_array==1)[1],np.where(labled_array==1)[0],alpha=0.2)
#        plt.scatter(np.where(labled_array==2)[1],np.where(labled_array==2)[0],alpha=0.2)
#        plt.scatter(np.where(labled_array==3)[1],np.where(labled_array==3)[0],alpha=0.2)
#        plt.imshow(xyz_blurred)

    objects=scim.find_objects(labled_array)
    
    object_ratios=[]
    
    for obj in objects:
        if labled_array[obj].size>30:
            object_ratios.append(len(labled_array[obj])/len(labled_array[obj][0]))
        else:object_ratios.append(100)
        
    spot_slice = objects[(np.abs(np.array(object_ratios)-1)).argmin()]
    
    #boundingbox=np.where(xyz_blurred[spot_slice] > threshold_value)#+spot_slice[0]
    #plt.scatter(boundingbox[1]+spot_slice[1].start,boundingbox[0]+spot_slice[0].start,color='C3')

    xyz_blurred=gaussian_filter(xyz, sigma=len(xyz_blurred[spot_slice][0])/10)
    
    yo,xo=scim.center_of_mass(np.clip(xyz_blurred[spot_slice],threshold_value,amplitude)-threshold_value)

    #plt.plot(xo,yo,marker='o')
    
    #plt.imshow(xyz_blurred)
    
    #plt.show()
    
    #plt.clf()
    
    yo+=spot_slice[0].start
    xo+=spot_slice[1].start
    
    if flag == True:
        threshold_value=(amplitude-offset)*0.50+offset
    
    sigma_x,sigma_y,theta=find_axis(xyz_blurred,int(np.round(xo)),int(np.round(yo)),threshold_value)
    
    #print(amplitude, xo, yo, sigma_x, sigma_y, theta, offset)
    
    return (amplitude, xo, yo, sigma_x/2.35, sigma_y/2.35, theta, offset)
    
def bounds(ig):
    
    lower_factor=0.5
    upper_factor=1.5
    
    amplitude_l=ig[0]*lower_factor
    sigma_x_l=ig[3]*lower_factor
    sigma_y_l=ig[4]*lower_factor
    xo_l=ig[1]-ig[3]*(1-lower_factor)
    yo_l=ig[2]-ig[4]*(1-lower_factor)
    theta_l=ig[5]-sc.pi/6
    offset_l=ig[6]*lower_factor
    
    amplitude=ig[0]*upper_factor
    sigma_x=ig[3]*upper_factor
    sigma_y=ig[4]*upper_factor
    xo=ig[1]+ig[3]*(upper_factor-1)
    yo=ig[2]+ig[4]*(upper_factor-1)
    theta=ig[5]+sc.pi/6
    offset=ig[6]*upper_factor
    
    return [amplitude_l, xo_l, yo_l, sigma_x_l, sigma_y_l, theta_l, offset_l],[amplitude, xo, yo, sigma_x, sigma_y, theta, offset]

def crop_shots(z,ig):
    y0=int(ig[1])#len(z)/2#
    x0=int(ig[2])#len(z[0])/2#
    width=max(ig[3:4])
    zx=len(z[0])
    zy=len(z)
    xlimit_l=np.clip(x0-int(3*width),0,zx)
    ylimit_l=np.clip(y0-int(3*width),0,zy)
    xlimit_h=np.clip(x0+int(3*width),0,zx)
    ylimit_h=np.clip(y0+int(3*width),0,zy)
    #print(y0,x0,zx,zy,xlimit_l,ylimit_l,xlimit_h,ylimit_h)
    return z[xlimit_l:xlimit_h,ylimit_l:ylimit_h]
   

def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = np.clip(amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2))),offset,5000)
    return g.ravel()

def twoD_Gaussian_mesh(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = np.clip(amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2))),offset,1E20)
    return g

class bias_group:
    
    def __init__(self, voltage_change, V, shot):
        if voltage_change==0:
            self.bias = V
            self.shot_count=1
            self.x_section=shot
        else:
            self.bias = V
            self.shot_count=0
            self.x_section=shot-shot
        
    def add_shot(self, voltage_change, V, shot):
        if voltage_change==0:
            self.bias+=V
            self.x_section=self.x_section+shot
            self.shot_count+=1
    
    def fit_gaussain(self, bg):
        
        self.x_section=np.clip(self.x_section-bg*self.shot_count,0,1E10)
        
        self.ig=initial_guess(self.x_section)
        self.crop=crop_shots(self.x_section,self.ig)
        self.ig_ave=initial_guess(self.crop,flag=True)
        
        self.x = np.linspace(0,len(self.crop),len(self.crop))
        self.y = np.linspace(0,len(self.crop[0]),len(self.crop[0]))
        self.x,self.y = np.meshgrid(self.x, self.y)
        self.xy=self.x,self.y
        
        self.popt, self.pcov = so.curve_fit(twoD_Gaussian, self.xy, self.crop.reshape(len(self.crop)*len(self.crop[0])), p0=self.ig_ave, bounds=bounds(self.ig_ave))
        self.stdx=self.popt[3] 
        self.stdy=self.popt[4]
        self.x0=self.popt[1]#+ig[1]-ig[3])
        self.y0=self.popt[2]#+ig[2]-ig[3])
        self.stderr=np.diag(self.pcov)
        self.estdx=self.stderr[3]
        self.estdy=self.stderr[4]
        self.ig_stdx=self.ig_ave[1]
        self.ig_stdy=self.ig_ave[2]
        self.sumpix=sum(sum(self.crop))
        self.theta=self.popt[5]
        self.has_fit=True
        
        self.fit=[self.bias/self.shot_count,self.stdx,self.stdy,self.estdx,self.estdy,self.x0,self.y0,self.theta]
    
    def save_fig(self, k):
        if self.has_fit == True:
            self.Z=twoD_Gaussian_mesh(self.xy,*self.ig_ave)
            self.Z1=twoD_Gaussian_mesh(self.xy,*self.popt)
            plt.contour(self.x, self.y, self.Z,linewidths=0.5,colors='C4')
            plt.contour(self.x, self.y, self.Z1,linewidths=0.5,colors='C1')
            plt.imshow(self.crop, cmap=plt.cm.viridis, interpolation='nearest')
        else: plt.imshow(self.x_section, cmap=plt.cm.viridis, interpolation='nearest')
        
        plt.colorbar()
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.title(self.bias/self.shot_count)
        
        if not os.path.exists('img'):
            os.makedirs('img')
            
        plt.savefig('img/{}.png'.format(k),dpi=250)
        plt.clf()