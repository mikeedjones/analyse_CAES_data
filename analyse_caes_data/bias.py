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
#        plt.xticks([])
#        plt.yticks([])
#        plt.xlim([0,89])
#        plt.ylim([0,90])
        
#    plt.savefig('im_with_axes.pdf')
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
        
#        plt.plot(thetas,find_axis,'.')
#        plt.plot(thetas,fit_cos(thetas, *psin))
#        plt.xlabel('$\\theta$')
#        plt.xticks(np.linspace(0,2*sc.pi,5),['0','$\\pi/2$','$\\pi$','$3\\pi/4$','$2\\pi$'])
#        plt.ylabel('Sum along line at $\\theta$')
#        plt.savefig('guess_rotation.pdf')

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
        offset=0
    else:
        xyz=gaussian_filter(xyz, sigma=len(xyz[0])/10)
        amplitude=np.nanmax(xyz.flatten())
        offset=0
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
    
    return [1, xo, yo, 1+sigma_x/2.35, 1+sigma_y/2.35, theta+sc.pi/2, 0]

    
def bounds(ig):
    
    lower_factor=0.6
    upper_factor=1.8
    
    amplitude_l=ig[0]*lower_factor
    sigma_x_l=ig[3]*lower_factor
    sigma_y_l=ig[4]*lower_factor
    sigma_x=ig[3]*upper_factor
    sigma_y=ig[4]*upper_factor
    xo_l=ig[1]-sigma_x-5
    yo_l=ig[2]-sigma_y-5
    theta_l=ig[5]-sc.pi/6
    offset_l=-1
    
    amplitude=ig[0]*upper_factor
    xo=ig[1]+sigma_x+5
    yo=ig[2]+sigma_y+5
    theta=ig[5]+sc.pi/6
    offset=1
    
#    print(np.array([amplitude_l, xo_l, yo_l, sigma_x_l, sigma_y_l, theta_l, offset_l]))
#    print(np.array([amplitude, xo, yo, sigma_x, sigma_y, theta, offset]))    
    
    return [amplitude_l, xo_l, yo_l, sigma_x_l, sigma_y_l, theta_l, min([offset
                                                                         ,offset_l])],[amplitude, xo, yo, sigma_x, sigma_y, theta, max([offset_l,offset])]

def bounds_1d(ig,yflag=False):
    
    lower_factor=0.75
    upper_factor=1.5
    
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
    
    lower_factor=0.8
    upper_factor=1.2
    
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
    
    fac=4
    
    xlimit_l=np.clip(x0-int(fac*width),0,zy)
    ylimit_l=np.clip(y0-int(fac*width),0,zx)
    xlimit_h=np.clip(x0+int(fac*width),0,zy)
    ylimit_h=np.clip(y0+int(fac*width),0,zx)
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

def rotate(im,x0,y0):
    X=len(im[1])
    Y=len(im)

    for N in np.linspace(0,360,20)[:-1]:
        im_t=sim.rotate(im,N)
        
        x_corner=int(np.clip(len(im_t[1])/2-x0,0,1000))
        y_corner=int(np.clip(len(im_t)/2-y0,0,1000))
        
        im_t=im_t[x_corner:x_corner+X,y_corner:y_corner+Y]
        try:
            im=im+im_t
        except:
            continue
    
    return im
    
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
    
    def add_locs(self, locs):
        self.locs=locs[:,-1,:]
        im=self.x_section
        hist=np.histogram2d(*self.locs*1e3*pixpermm,bins=[len(im),len(im[1])],range=np.array([[-len(im),len(im)],[-len(im[1]),len(im[1])]])/2)

        self.loc_group=bias_group(0, self.bias, hist[0])
        self.loc_group.single_shot(0)
        self.loc_group.locs_fit(self.locs)
            
    def single_shot(self,bg):
        for section in self.x_sections:
            self.base=section-bg
            self.base=self.base-np.mean(self.base[:10,:10])
        self.e_count=np.sum(self.base)/21.7
        self.x_section_ub=(self.base)/np.max(self.base)
        
        self.initial_guess(bg)
        
        self.x_section=(self.x_section/np.max(self.x_section))
        
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
#        if obj_count>4:        
        self.blur=self.ig[3]/2
#            print(obj_count)
        self.crop=gaussian_filter(self.crop, sigma=self.blur,mode='constant',cval=0)
        
        self.crop=self.crop/np.max(self.crop)
        
        self.ig_ave=initial_guess(self.crop,flag=True,ig=self.ig)
        
        self.crop=rotate(self.crop_ub,*self.ig_ave[1:3])
        
        self.crop=self.crop/np.max(self.crop)
        
        self.blur=0
        
        self.ig_ave=initial_guess(self.crop,flag=True,ig=self.ig)
        
        self.crop=self.crop_ub
        
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
        
        self.fit1d=[self.bias/self.shot_count,self.shot_count,self.stdx1,self.stdy1,self.estd1x,self.estd1y,self.x01,self.y01,self.theta1,self.m_stdx,self.e_count,self.z0/self.shot_count,self.errz0/self.shot_count]
    
    def fit_gaussain(self, bg):
        self.popt, self.pcov = so.curve_fit(twoD_Gaussian, self.xy, self.crop.ravel(), p0=self.ig_ave, bounds=bounds(self.ig_ave))
        self.popt[3]=np.sqrt(self.popt[3]**2-2*self.blur**2)
        self.popt[4]=np.sqrt(self.popt[4]**2-2*self.blur**2)
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

    def locs_fit(self,locs):
        self.locs=locs
        d=2
        self.mean=d/(1E3*pixpermm)
        x_sec=locs[0][np.where((abs(locs[1])-self.mean<0))]
        self.stdx=np.std(x_sec)*1e3*pixpermm#np.sqrt(locs[1]**2+locs[1]**2)
        self.stdy=self.stdx
        self.x0=self.ig_ave[1]
        self.y0=self.ig_ave[1]
#        self.sumpix=sum(sum(self.crop))
        self.theta=0     
        self.has_fit=True
        
        self.popt=[1,self.x0,self.y0,self.stdx,self.stdy,self.theta,0]

    def locs_trans(self,ax):
        d=2/(1E3*pixpermm)
        n, bins, rects=ax.hist(self.locs[0][np.where((abs(self.locs[1])-d<0))]*1e3*pixpermm,density=True,bins=31,range=[-len(self.x_section),len(self.x_section)])
        h=[]
        for r in rects:
            h.append(r.get_height())
        
        for r in rects:
            r.set_height(r.get_height()/max(h))

#        ax.set_xlabel('x (mm)')
#        ax.set_ylabel('Density')
        
    def transverse(self, k=None,ax=plt.subplots(2,1)[1],ylab=True):
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

        #Show the x-cross sectio and the fit

        xlim=len(self.crop)/2
        ylim=len(self.crop[1])/2
        
        if self.has_fit1d == True:
            ax[0].plot(self.x_l1d,gauss_function_1d(self.x_l1d, *self.popt_tx[:-1],0),color='C4')
            ax[0].plot(self.x_l1d,self.x_data,color='C3')

        if self.has_fit==True:
            err_2d_gauss=[self.popt[0],0,self.popt[4],0]
            ax[0].plot(self.x_l1d,gauss_function_1d(self.x_l1d, *err_2d_gauss ),color='C1')
            ax[0].plot(self.x_l1d,self.x_data,color='C2')
        else:
            err_2d_gauss=[self.ig_ave[0],0,self.ig_ave[4],0]
            ax[0].plot(self.x_l1d,gauss_function_1d(self.x_l1d, *err_2d_gauss ),color='C3')
            ax[0].plot(self.x_l1d,self.x_data)
        
        ax[0].set_xlim(-xlim,xlim)
        ax[0].set_xlabel("x (mm)")
        ticks = ticker.FuncFormatter(lambda x, pos: '{0:.1f}'.format(x/pixpermm))
        ax[0].xaxis.set_major_formatter(ticks)
        if ylab:
            ax[0].set_ylabel('Intensity (a.u.)')
        if k!= None:
            ax[0].set_title(self.bias/self.shot_count)
        
         #Show the y-cross sectio and the fit       
        
        if self.has_fit1d == True:
            ax[1].plot(self.y_l1d,gauss_function_1d(self.y_l1d, *self.popt_ty[:-1],0),color='C4')
            ax[1].plot(self.y_l1d,self.y_data,color='C3')
        if self.has_fit==True:
            err_2d_gauss=[self.popt[0],0,self.popt[3],0]
            ax[1].plot(self.y_l1d,gauss_function_1d(self.y_l1d, *err_2d_gauss ),color='C1')
            ax[1].plot(self.y_l1d,self.y_data,color='C2')
        else:
            err_2d_gauss=[self.ig_ave[0],0,self.ig_ave[3],0]
            ax[1].plot(self.y_l1d,gauss_function_1d(self.y_l1d, *err_2d_gauss ),color='C3')
            ax[1].plot(self.y_l1d,self.y_data)
        if ylab:
            ax[1].set_ylabel('Intensity (a.u.)')
        ax[1].set_xlabel("y (mm)")
        ax[1].set_xlim(-ylim,ylim)
        ticks = ticker.FuncFormatter(lambda x, pos: '{0:.1f}'.format(x/pixpermm))
        ax[1].xaxis.set_major_formatter(ticks)
        
        if not os.path.exists('imgt'):
            os.makedirs('imgt')
            
        if k!= None:
            plt.savefig('imgt/{}.png'.format(k),dpi=100)
            plt.clf()
        

        
    def save_fig(self, k=None, ax=plt.gca(),ylab=True):
        xlim=len(self.crop_ub[1])
        ylim=len(self.crop_ub)
        if self.has_fit == True:
            self.Z_guess=twoD_Gaussian_mesh(self.xy,*self.ig_ave)
            self.Z_fit=twoD_Gaussian_mesh(self.xy,*self.popt)
#            plt.contour(self.x, self.y, self.Z_guess,linewidths=0.5,colors='C4')
            ax.contour((self.x), (self.y), self.Z_fit,5,linewidths=0.5,colors='C1')
            ax.imshow(self.crop_ub, cmap=plt.cm.viridis, interpolation='nearest',vmin=0,vmax=1)#,extent=np.array([-xlim,xlim,-ylim,ylim])/pixpermm)
        else: 
            self.Z=twoD_Gaussian_mesh(self.xy,*self.ig_ave)
            ax.contour(self.x, self.y, self.Z,linewidths=0.5,colors='C4')
            ax.imshow(self.crop_ub, cmap=plt.cm.viridis, interpolation='nearest',vmin=0,vmax=1)#,extent=[-xlim,-ylim,xlim,ylim])

        ticksx = ticker.FuncFormatter(lambda x, pos: '{0:.1f}'.format((x-self.x0)/pixpermm))
        ax.set_xticks([6,xlim/2,xlim-6])
        ax.xaxis.set_major_formatter(ticksx)
        
        ticksy = ticker.FuncFormatter(lambda y, pos: '{0:.1f}'.format((y-self.y0)/pixpermm))
        ax.set_yticks([6,ylim/2,ylim-6])
        ax.yaxis.set_major_formatter(ticksy)
#        plt.colorbar()
        ax.set_xlabel("x (mm)")
        if ylab:
            ax.set_ylabel("y (mm)")
        
        if not os.path.exists('img'):
            os.makedirs('img')
        
        if k!= None:
            ax.set_title('{:.1f}'.format(self.bias/self.shot_count))
            plt.savefig('img/{}.png'.format(k),dpi=100)
            plt.clf()