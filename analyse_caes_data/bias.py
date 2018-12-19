# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 14:07:27 2018

bias class to contain all the data from the shots taken at that bias and perform calculations on them

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

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import label

'''Estimates the width and rotation of the gaussain by linear fitting the gaussain and then summing the
number of points above threshold along that line. Then that line is rotated by pi/2 and the process is repeated
for the second axis.'''

pixpermm=53.4
#blur_fact=15

def fit_cos(x,a,b, phi):
    x=2*(np.array(x)+phi)
    return a*np.cos(x)+b

def axis_lengths(xyz, xo,yo,threshold_value,theta,xa=np.array([-1000,1000])):
    l=[]
    ax=[]
    line_datas=[]
    ra=np.linspace(-200,200,1000)
#    plt.imshow(xyz)
    for th in [theta, theta+sc.pi/2]:
        xt=ra*np.cos(th)+xo
        yt=ra*np.sin(th)+yo

        line_data=sim.map_coordinates(xyz,[yt,xt])
        line_datas.append(line_data)
        l.append(sum(np.where(line_data> threshold_value, 1, 0))*(ra[1]-ra[0]))
        ax.append(ra)
#        plt.plot(xt,yt)
#    plt.show()
#    plt.clf() 
    
    return l, ax, line_datas

def find_axis(xyz,xo,yo,threshold_value,theta=False):
    ra=np.linspace(-200,200,2000)
    find_axis=[]
    thetas=[]
    if theta==False:
        for theta in np.linspace(0,2*sc.pi,100):
            xt=ra*np.cos(theta)+xo
            yt=ra*np.sin(theta)+yo
            line_data=sim.map_coordinates(xyz,[yt,xt])
            find_axis.append(sum(line_data))
            thetas.append(theta)
    
        mean=np.mean(find_axis)
        try:
            psin,psincov=so.curve_fit(fit_cos,thetas,find_axis,p0=[max(find_axis)-mean,mean,thetas[np.argmin(find_axis[:50])]],bounds=[[(max(find_axis)-mean)*0.9,mean*0.9, -sc.pi],[(max(find_axis)-mean)*1.1,mean*1.1, sc.pi]]) 
            theta=psin[2]
        except:
            theta=thetas[np.argmax(find_axis[:50])]
#        plt.plot(thetas,find_axis)
#        plt.plot(thetas,fit_cos(thetas, *psin))
#        plt.show()
#        plt.clf()

    
    l,ax,line_data=axis_lengths(xyz, xo,yo,threshold_value,theta)
      
  
    return l[0], l[1], theta

def find_amplitudes(xyz):
    a=[]
    maxA=np.nanmax(xyz.flatten())
    for th in np.linspace(maxA,0,100):
        a.append(sum(sum(np.clip(xyz,th,maxA)-th)))
    
#    plt.plot(np.linspace(0,maxA,100),a)
#    plt.show()
#    plt.clf()
    
#    return amp0, amp1
    

def gauss_function_1d(x, a, x0, sigma,offset):
    return np.clip(a*np.exp(-(x-x0)**2/(2*sigma**2))+offset,0,1E10)  

def overlapped_1d(x, a, a1, x0, sigma, sigma1, offset):
    
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+a1*np.exp(-(x-x0)**2/(2*sigma1**2))+offset

def overlap_initial_guess(xyz,threshold_perc=95,flag=False):
    if flag == True:
        amplitude=np.nanmax(xyz.flatten())  
        threshold_value1=amplitude*0.3
        threshold_value=amplitude*0.8
        offset=amplitude/10  
    else:
        xyz=gaussian_filter(xyz, sigma=len(xyz[0])/10)
        amplitude=np.nanmax(xyz.flatten())
        offset=amplitude/10
        threshold_value=threshold_value1=amplitude*0.25
    
    yo,xo=scim.center_of_mass(np.clip(xyz,threshold_value,amplitude)-threshold_value)
    
    sigma_y,sigma_x,theta=find_axis(xyz,xo,yo,threshold_value)
    sigma_y1,sigma_x1,theta=find_axis(xyz,xo,yo,threshold_value1,theta=theta)

    
    return (1*amplitude/2, 1*amplitude/2, xo, yo, sigma_x/2.35, sigma_y/2.35, (sigma_x1/2.35), (sigma_y1/2.35), theta, offset)


def initial_guess(xyz,threshold_perc=95,flag=False,ig=[0,0,0,0,0]):
    if flag == True:
        amplitude=np.nanmax(xyz.flatten())  
        threshold_value=amplitude*0.5
    else:
        sigma_blur=len(xyz[0])/20
#        plt.subplot(121)
#        plt.imshow(xyz)
        xyz=gaussian_filter(xyz, sigma=sigma_blur)
#        plt.subplot(122)
#        plt.imshow(xyz)
#        plt.show()
#        plt.clf()
        amplitude=np.nanmax(xyz.flatten())
        threshold_value=amplitude*0.5

    yo,xo=scim.center_of_mass(np.clip(xyz,threshold_value,amplitude)-threshold_value)
    
    sigma_y,sigma_x,theta=find_axis(xyz,xo,yo,threshold_value)

#    print([amplitude, xo, yo, sigma_x, sigma_y, theta, amplitude/2])
    
    return [amplitude, xo, yo, 1+sigma_x/2.35, 1+sigma_y/2.35, theta+sc.pi/2, 0]

    
def bounds(ig):
    
    lower_factor=0.25
    upper_factor=4
    
    amplitude_l=ig[0]*lower_factor
    sigma_x_l=ig[3]*lower_factor
    sigma_y_l=ig[4]*lower_factor
    sigma_x=ig[3]*upper_factor
    sigma_y=ig[4]*upper_factor
    xo_l=ig[1]-sigma_x-1
    yo_l=ig[2]-sigma_y-1
    theta_l=ig[5]-sc.pi/6
    offset_l=-1
    
    amplitude=ig[0]*upper_factor
    xo=ig[1]+sigma_x+1
    yo=ig[2]+sigma_y+1
    theta=ig[5]+sc.pi/6
    offset=1
    
#    print(np.array([amplitude_l, xo_l, yo_l, sigma_x_l, sigma_y_l, theta_l, offset_l]))
#    print(np.array([amplitude, xo, yo, sigma_x, sigma_y, theta, offset]))    
    
    return [amplitude_l, xo_l, yo_l, sigma_x_l, sigma_y_l, theta_l, min([offset
                                                                         ,offset_l])],[amplitude, xo, yo, sigma_x, sigma_y, theta, max([offset_l,offset])]

def bounds_1d(ig,yflag=False):
    
    lower_factor=0.25
    upper_factor=2
    
    amplitude_l=ig[0]*lower_factor
    sigma_x_l=ig[3]*lower_factor
    xo_l=ig[1]-ig[3]*(upper_factor)
    offset_l=ig[6]*lower_factor
    
    amplitude=ig[0]*upper_factor
    sigma_x=ig[3]*upper_factor
    xo=ig[1]+ig[3]*(upper_factor)
    offset=ig[6]*upper_factor
    
    if yflag==True:
        xo_l=ig[2]-ig[4]*(upper_factor)
        xo=ig[2]+ig[4]*(upper_factor)
    
    #print(np.array([amplitude_l, xo_l, yo_l, sigma_x_l, sigma_y_l, theta_l, offset_l])-np.array([amplitude, xo, yo, sigma_x, sigma_y, theta, offset]))
    
    return [amplitude_l, xo_l, sigma_x_l, min([offset_l,offset]),0],[amplitude, xo, sigma_x, max([offset_l,offset]),1E5]

def bounds_2(ig,yflag=False):
    
    lower_factor=0.75
    upper_factor=1.5
    
    amplitude_l=ig[0]*lower_factor
    amplitude1_l=ig[1]*lower_factor
    sigma_x_l=ig[4]*lower_factor
    sigma_y_l=ig[5]*lower_factor
    sigma_x1_l=ig[6]*(lower_factor**2)
    sigma_y1_l=ig[7]*(lower_factor**2)
    xo_l=ig[2]-abs(ig[4])
    yo_l=ig[3]-abs(ig[5])
    theta_l=ig[8]-sc.pi/10
    offset_l=ig[9]*lower_factor
    
    amplitude=ig[0]*upper_factor
    amplitude1=ig[1]*upper_factor
    sigma_x=ig[4]*upper_factor
    sigma_y=ig[5]*upper_factor
    sigma_x1=ig[6]*(upper_factor**2)
    sigma_y1=ig[7]*(upper_factor**2)
    xo=ig[2]+abs(ig[4])
    yo=ig[3]+abs(ig[5])
    theta=ig[8]+sc.pi/10
    offset=ig[9]*upper_factor
    
#    print(np.array([amplitude_l, xo_l, yo_l, sigma_x_l, sigma_y_l, theta_l, offset_l]))
#    print(np.array([amplitude, xo, yo, sigma_x, sigma_y, theta, offset]))
    
    return [amplitude_l,amplitude1_l, xo_l, yo_l, sigma_x_l, sigma_y_l,sigma_x1_l, sigma_y1_l, theta_l, min([offset_l,offset])],[amplitude,amplitude1, xo, yo, sigma_x, sigma_y,sigma_x1, sigma_y1, theta, max([offset_l,offset])]

def crop_shots(z,ig):
    y0=int(ig[1])#len(z)/2#
    x0=int(ig[2])#len(z[0])/2#
    width=max(ig[3:4])
    zx=len(z[0])
    zy=len(z)
    xlimit_l=np.clip(x0-int(4*width),0,zy)
    ylimit_l=np.clip(y0-int(4*width),0,zx)
    xlimit_h=np.clip(x0+int(4*width),0,zy)
    ylimit_h=np.clip(y0+int(4*width),0,zx)
    #print(y0,x0,zx,zy,xlimit_l,ylimit_l,xlimit_h,ylimit_h)
    return z[xlimit_l:xlimit_h,ylimit_l:ylimit_h]

def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo) 

    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))+offset
    return g.ravel()

def twoD_Gaussian_mesh(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo) 

    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))+offset
    return g

def overlap_twoD_Gaussian(xy, amplitude, amplitude1, xo, yo, sigma_x, sigma_y,sigma_x1, sigma_y1, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo) 

    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    
    a1 = (np.cos(theta)**2)/(2*sigma_x1**2) + (np.sin(theta)**2)/(2*sigma_y1**2)
    b1 = -(np.sin(2*theta))/(4*sigma_x1**2) + (np.sin(2*theta))/(4*sigma_y1**2)
    c1 = (np.sin(theta)**2)/(2*sigma_x1**2) + (np.cos(theta)**2)/(2*sigma_y1**2)
    
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))+amplitude1*np.exp( - (a1*((x-xo)**2) + 2*b1*(x-xo)*(y-yo) 
                            + c1*((y-yo)**2)))+offset
    return g.ravel()

def overlap_twoD_Gaussian_mesh(xy, amplitude, amplitude1, xo, yo, sigma_x, sigma_y,sigma_x1, sigma_y1, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo) 

    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    
    a1 = (np.cos(theta)**2)/(2*sigma_x1**2) + (np.sin(theta)**2)/(2*sigma_y1**2)
    b1 = -(np.sin(2*theta))/(4*sigma_x1**2) + (np.sin(2*theta))/(4*sigma_y1**2)
    c1 = (np.sin(theta)**2)/(2*sigma_x1**2) + (np.cos(theta)**2)/(2*sigma_y1**2)
    
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))+amplitude1*np.exp( - (a1*((x-xo)**2) + 2*b1*(x-xo)*(y-yo) 
                            + c1*((y-yo)**2)))+offset
    return g

def add_array_offset(b1,b2,offset,center_h,center_v):

    pos_v, pos_h = offset  # offset
    v_range1 = slice(max(0, pos_v), max(min(pos_v + b2.shape[0], b1.shape[0]), 0))
    h_range1 = slice(max(0, pos_h), max(min(pos_h + b2.shape[1], b1.shape[1]), 0))
    
    v_range2 = slice(max(0, -pos_v), min(-pos_v + b1.shape[0], b2.shape[0]))
    h_range2 = slice(max(0, -pos_h), min(-pos_h + b1.shape[1], b2.shape[1]))
    
    b1[v_range1, h_range1] += b2[v_range2, h_range2]
    
#    plt.imshow(b1)
#    plt.scatter(pos_h+center_h,pos_v+center_v)
#    plt.scatter(b1.shape[0]/2,b1.shape[1]/2)
##    plt.scatter(xo,yo)
#    plt.show()

    return b1

class bias_group:
    
    def __init__(self, voltage_change, V, shot, modelstdx=0,z0=0,errz0=0):
        if voltage_change==0:
            self.bias = V
            self.shot_count=1
            self.x_section=shot
            self.m_stdx=modelstdx
            self.VC=voltage_change
            self.z0=z0
            self.errz0=errz0
            
            self.has_fit=False
            self.has_fit1d=False
            
            self.x_sections=[shot]
            #self.igs=[initial_guess(self.x_section,flag=True)]
        
        else:
            self.bias = V
            self.shot_count=0
            self.x_section=shot-shot
            self.m_stdx=modelstdx
            self.VC=voltage_change
            self.z0=z0
            self.errz0=errz0
            
            self.has_fit=False
            self.has_fit1d=False
        
    def add_shot(self, voltage_change, V, shot, modelstdx=0,z0=0,errz0=0):
        if voltage_change==0:
            self.bias+=V
            self.shot_count+=1
            self.m_stdx=modelstdx
            self.VC=voltage_change
            self.x_sections.append(shot)
            self.z0+=z0
            self.errz0+=errz0
            
    def single_shot(self,bg):
        for section in self.x_sections:
            self.base=section-bg
        self.e_count=(sum(sum(self.base))/364.994)
        off=np.mean(self.base[:10,:10])
        self.x_section_ub=(self.base-off)/max(self.base.flatten()-off)
        
        self.initial_guess(bg)
        
        f=self.x_section.flatten()
        self.x_section=(self.x_section/(max(f)))
        self.e_count=0
        
    def align_shots(self,bg):
        l=500
        self.base=np.zeros([l,l])
        self.e_count=[]
        for section in self.x_sections:
            section=np.clip(section-bg,0,1E10)
            section_b=gaussian_filter(section, sigma=len(section[0])/50)
            
#            plt.imshow(section_b)
#            plt.colorbar()
##            plt.scatter(xo,yo)
#            plt.show()

#            low_values_flags = shot < 50  # Where values are low
#            shot[low_values_flags] = 0
            flat=section_b.flatten()
            amplitude=max(flat)
            threshold_value=amplitude*0.5
            temp=np.clip(section_b,threshold_value,amplitude)-threshold_value
            
            yo,xo=scim.center_of_mass(temp)
            
            offx=int(self.base.shape[1]/2-xo)
            offy=int(self.base.shape[0]/2-yo)
            offset= offy, offx
            self.e_count.append(sum(flat))
            self.base=add_array_offset(self.base,np.clip(section,0,1E6),offset,yo,xo)
#        print('xo={}, yo={}'.format(xo,yo))
#        plt.imshow(self.base)
##        plt.plot(xo,yo)
#        plt.colorbar()
#        plt.show()
        self.x_section=self.base
        self.e_count=np.mean(self.e_count)


    def initial_guess(self, bg):
        self.ig=initial_guess(self.x_section_ub)
        self.crop=crop_shots(self.x_section_ub,self.ig)
        self.crop_ub=crop_shots(self.x_section_ub,self.ig)
        
        connective_structure=[[1,1,1],
                          [1,1,1],
                          [1,1,1]]
        obj_count=label(np.clip(self.crop,self.ig[0]*0.6,self.ig[0])-self.ig[0]*0.6,structure=connective_structure)[1]
        self.blur=0#self.ig[3]
        if obj_count>10:        
            self.blur=self.ig[3]
#            print(obj_count)
            self.crop=gaussian_filter(self.crop, sigma=self.blur,mode='constant',cval=0)
        
        
        self.ig_ave=initial_guess(self.crop,flag=True,ig=self.ig)
        self.x_l = np.linspace(0,len(self.crop[0])-1,len(self.crop[0]))
        self.y_l = np.linspace(0,len(self.crop)-1,len(self.crop))
        self.x,self.y = np.meshgrid(self.x_l, self.y_l)
        self.xy=self.x,self.y
        
        self.x0_ig=self.ig_ave[1]#+ig[1]-ig[3])
        self.y0_ig=self.ig_ave[2]#+ig[2]-ig[3])
        
        
    def fit_1d_gaussain(self, bg):
        
        self.amin=self.ig_ave[0]/2
        
        self.popt_tx, self.pcov_tx = so.curve_fit(gauss_function_1d, self.x_l1d, self.x_data, p0=[self.ig_ave[0],self.ig_ave[2],self.ig_ave[4],self.ig_ave[6],self.amin],bounds=bounds_1d(self.ig_ave,yflag=True))
        self.popt_ty, self.pcov_ty = so.curve_fit(gauss_function_1d, self.y_l1d, self.y_data, p0=[self.ig_ave[0],self.ig_ave[1],self.ig_ave[3],self.ig_ave[6],self.amin],bounds=bounds_1d(self.ig_ave))

        self.stdx1=self.popt_tx[2] 
        self.stdy1=self.popt_ty[2]
        self.x01=self.popt_tx[1]#+ig[1]-ig[3])
        self.y01=self.popt_ty[1]#+ig[2]-ig[3])
        
        self.stderr1x=np.diag(self.pcov_tx)
        self.stderr1y=np.diag(self.pcov_ty)
        self.estd1x=self.stderr1x[2]
        self.estd1y=self.stderr1y[2]
        self.ig_stdx1=self.ig_ave[1]
        self.ig_stdy1=self.ig_ave[2]
        self.sumpix1=sum(sum(self.crop))      
        self.has_fit1d=True
        self.theta1=0
        
        self.e_count=(sum(sum(self.x_section))/364.994)
        
        self.fit1d=[self.bias/self.shot_count,self.shot_count,self.stdx1,self.stdy1,self.estd1x,self.estd1y,self.x01,self.y01,self.theta1,self.m_stdx,self.e_count,self.z0/self.shot_count,self.errz0/self.shot_count]
    
    def fit_gaussain(self, bg):
        self.popt, self.pcov = so.curve_fit(twoD_Gaussian, self.xy, self.crop.ravel(), p0=self.ig_ave, bounds=bounds(self.ig_ave))
        self.popt[3]=np.sqrt(self.popt[3]**2-self.blur**2)
        self.popt[4]=np.sqrt(self.popt[4]**2-self.blur**2)
        self.popt[0]=1
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
        
        self.fit=[self.bias/self.shot_count,self.shot_count,self.stdx,self.stdy,self.estdx,self.estdy,self.x0,self.y0,self.theta,self.stdx-self.popt[6],self.e_count,self.z0/self.shot_count,self.errz0/self.shot_count]
        
    def transverse(self, k):
        if self.has_fit == True:
            l,transverse,line_data=axis_lengths(self.crop_ub, self.popt[1],self.popt[2],0,self.popt[5])
            self.x_data=line_data[1]
            self.y_data=line_data[0]
            self.x_l1d =transverse[0]
            self.y_l1d =transverse[1]
        else:
            l,transverse,line_data=axis_lengths(self.crop_ub, self.ig_ave[1],self.ig_ave[2],0,self.ig_ave[5])
            self.x_data=line_data[1]
            self.y_data=line_data[0]
            self.x_l1d =transverse[0]
            self.y_l1d =transverse[1]
        
        plt.subplot(2,1,1)
        if self.has_fit1d == True:
            plt.plot(self.x_l1d,gauss_function_1d(self.x_l1d, *self.popt_tx[:-1],0))
            plt.plot(self.x_l1d,self.x_data)

        if self.has_fit==True:
#            over_2d_gauss=[self.popt[0],self.popt[1],0,self.popt[5],self.popt[7],self.popt[9]]
            err_2d_gauss=[self.popt[0],0,self.popt[4],0]
#            plt.plot(self.x_l1d,overlapped_1d(self.x_l1d, *over_2d_gauss ),color='C4')
            plt.plot(self.x_l1d,gauss_function_1d(self.x_l1d, *err_2d_gauss ),color='C4')
            plt.plot(self.x_l1d,self.x_data)
        else:
#            over_2d_gauss=[self.ig_ave[0],self.ig_ave[1],0,self.ig_ave[4],self.ig_ave[6],self.ig_ave[9]]
            err_2d_gauss=[self.ig_ave[0],0,self.ig_ave[4],0]
#            plt.plot(self.x_l1d,overlapped_1d(self.x_l1d, *over_2d_gauss ),color='C3')
            plt.plot(self.x_l1d,gauss_function_1d(self.x_l1d, *err_2d_gauss ),color='C3')
            plt.plot(self.x_l1d,self.x_data)
        
        plt.xlim(-1.5*pixpermm,1.5*pixpermm)
        ticks = ticker.FuncFormatter(lambda x, pos: '{0:.1f}'.format(x/pixpermm))
        plt.gca().xaxis.set_major_formatter(ticks)
        plt.ylabel('Intensity (a.u.)')
        plt.title(self.bias/self.shot_count)
        
        plt.subplot(2,1,2)
        if self.has_fit1d == True:
            plt.plot(self.y_l1d,gauss_function_1d(self.y_l1d, *self.popt_ty[:-1],0))
            plt.plot(self.y_l1d,self.y_data)
        if self.has_fit==True:
#            over_2d_gauss=[self.popt[0],self.popt[1],0,self.popt[4],self.popt[6],self.popt[9]]
            err_2d_gauss=[self.popt[0],0,self.popt[3],0]
#            plt.plot(self.y_l1d,overlapped_1d(self.y_l1d, *over_2d_gauss ),color='C4')
            plt.plot(self.y_l1d,gauss_function_1d(self.y_l1d, *err_2d_gauss ),color='C4')
            plt.plot(self.y_l1d,self.y_data)
        else:
#            over_2d_gauss=[self.ig_ave[0],self.ig_ave[1],0,self.ig_ave[5],self.ig_ave[7],self.ig_ave[9]]
            err_2d_gauss=[self.ig_ave[0],0,self.ig_ave[3],0]
#            plt.plot(self.y_l1d,overlapped_1d(self.y_l1d, *over_2d_gauss ),color='C3')
            plt.plot(self.y_l1d,gauss_function_1d(self.y_l1d, *err_2d_gauss ),color='C3')
            plt.plot(self.y_l1d,self.y_data)
        plt.ylabel('Intensity (a.u.)')
        plt.xlabel("x and y (mm)")
            #self.fit1d=[self.bias/self.shot_count,self.shot_count,self.stdx,self.stdy,self.estdx,self.estdy,self.x0,self.y0,self.theta,self.m_stdx,self.e_count,self.z0/self.shot_count,self.errz0/self.shot_count]
        plt.xlim(-1.5*pixpermm,1.5*pixpermm)
        ticks = ticker.FuncFormatter(lambda x, pos: '{0:.1f}'.format(x/pixpermm))
        plt.gca().xaxis.set_major_formatter(ticks)

        
        if not os.path.exists('imgt'):
            os.makedirs('imgt')
        
        plt.savefig('imgt/{}.png'.format(k),dpi=250)
        plt.clf()
        

        
    def save_fig(self, k):
        if self.has_fit == True:
            self.Z_guess=twoD_Gaussian_mesh(self.xy,*self.ig_ave)
            self.Z_fit=twoD_Gaussian_mesh(self.xy,*self.popt)
#            plt.contour(self.x, self.y, self.Z_guess,linewidths=0.5,colors='C4')
            plt.contour(self.x, self.y, self.Z_fit,linewidths=0.5,colors='C1')
            plt.imshow(self.crop_ub, cmap=plt.cm.viridis, interpolation='nearest',vmin=0,vmax=1)
        else: 
            self.Z=twoD_Gaussian_mesh(self.xy,*self.ig_ave)
            plt.contour(self.x, self.y, self.Z,linewidths=0.5,colors='C4')
            plt.imshow(self.crop_ub, cmap=plt.cm.viridis, interpolation='nearest',vmin=0,vmax=1)

        
        plt.colorbar()
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.title('{:.1f}'.format(self.bias/self.shot_count))
        
        if not os.path.exists('img'):
            os.makedirs('img')
            
        plt.savefig('img/{}.png'.format(k),dpi=250)
        plt.clf()