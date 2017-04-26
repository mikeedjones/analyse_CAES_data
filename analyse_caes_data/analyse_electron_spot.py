# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 07:43:06 2016

@author: Michael
"""
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
import scipy.ndimage.measurements as scim

def width_at(xyz,xo,yo,amp):
    x_max=np.where(xyz[:,yo] > amp/2, 1, 0)
    y_max=np.where(xyz[xo,:] > amp/2, 1, 0)
    sigma_x=sum(x_max)
    sigma_y=sum(y_max)
    
    return sigma_x, sigma_y

def initial_guess(xyz):
    amplitude=max(xyz.flatten())
    xo,yo=scim.center_of_mass(np.where(xyz > amplitude/2, 1, 0))
    sigma_x,sigma_y=width_at(xyz,int(xo),int(yo),amplitude)
    theta=0
    offset=0    
    
    return (amplitude, xo, yo, sigma_x, sigma_y, theta, offset)
    
def bounds(ig):
    amplitude=ig[0]*5
    xo=450
    yo=450
    sigma_x=ig[3]*3
    sigma_y=ig[4]*3
    theta=360
    offset=100
    
    return 0,[amplitude, xo, yo, sigma_x, sigma_y, theta, offset]

def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

def profileanimation(outfile):
    import subprocess
    subprocess.check_call(["/usr/local/bin/ffmpeg","-framerate", "10", "-i", "img/%1d.png", 
                              "-c:v", "libx264", "-r", "30", "-pix_fmt", "yuv420p", outfile])
    subprocess.check_call(["rm", "-rf",  "img"])

def generateBG(BGpath, verbose=0):
    BG=np.genfromtxt(BGpath,skip_header=1)
    return BG

def average_shots(names,lastnames,header,lastheader):
    if names.split()[4]==lastnames.split()[4] and header[1]==lastheader[1]:
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
    return z[int(ig[1])-100:int(ig[1])+100,int(ig[2])-100:int(ig[2])+100]

def importdata(directory, verbose=0):
    import os
    if not os.path.exists (directory):
        return -1
    i=0
    lastnames='x y z f 1000'
    lastheader=[0, 0]
    z_ave=np.zeros([200,200])
    x = np.linspace(0,200,200)
    y = np.linspace(0,200,200)
    x,y = np.meshgrid(x, y)
    xy=x,y
    j=1
    flag=True
    stdx=[]
    biases=[]
    jlist=[]
#    tx=np.round(np.linspace(0,950*34E-3,num=5))
#    txloc=tx/34E-3
#    
#    ty=np.round(np.linspace(0,600*34E-3,num=6))
#    tyloc=ty/34E-3+400
    expt_name=directory.split('/')[-1]    
    
    try:
        for root, dirs, files in os.walk(directory):
          for names in tqdm(files,smoothing=0):
            if names==".DS_Store" : continue   
            if verbose == 1:
              print('Hashing', names)
            filepath = os.path.join(root,names)
            
            xyz=np.transpose(np.genfromtxt(filepath,skip_header=1))
            header=np.genfromtxt(filepath, max_rows=1)
            
            flag=average_shots(names,lastnames,header,lastheader)
            
            ig=initial_guess(xyz)
            
            if flag == True:
                z_ave=align_shots(xyz,ig)+z_ave
                j+=1
            else:
                if j>3:
                    ig_ave=initial_guess(z_ave)  
                    plt.clf()
                    plt.imshow((np.clip(z_ave/ig_ave[0],0,1)), cmap=plt.cm.viridis, interpolation='nearest')
                    lam=float(lastnames.split()[4].split('.')[0])
                    bias=lastheader[1]
                    
                    plt.title(str(lam)+' nm, '+str(bias)+' V')
                    plt.colorbar()
    #                plt.xticks(txloc,tx)
    #                plt.yticks(tyloc,ty)
    #                plt.xlabel("x (mm)")
    #                plt.ylabel("y (mm)")
                    savpath = 'img'
                    
                    if not os.path.isdir(savpath):
                        os.makedirs(savpath)  
                    plt.savefig(savpath+'/{}.png'.format(i),dpi=250)
                    try:                              
                        popt, pcov = so.curve_fit(twoD_Gaussian, xy, z_ave.reshape(200**2), p0=ig_ave, bounds=bounds(ig_ave))
                        stdx.append(popt[3]) 
                        biases.append(bias)
                        jlist.append(j)
                    except:
                        print(names)
                        print(bias)
                        print(ig_ave)
                        print(bounds(ig))
                        import traceback
                        # Print the stack traceback
                        traceback.print_exc()
                    j=1
                    i+=1
                    z_ave=align_shots(xyz,ig)
            lastnames=names
            lastheader=header

                    
        
        os.remove(savpath+"/0.png")
        profileanimation(directory+"/"+expt_name+".mp4")
        plt.clf()
        plt.plot(biases,stdx,'x')
        plt.xlabel("Einzel lens bias / V")
        plt.ylabel("Electron bunch size / a.u")
        plt.savefig(directory+"/biases_vs_stdx.svg")
        return biases,stdx,jlist
    except:
        import traceback
        # Print the stack traceback
        traceback.print_exc()
        return biases,stdx,jlist
  


