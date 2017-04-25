# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 07:43:06 2016

@author: Michael
"""

import numpy as np
import matplotlib.pyplot as plt

def profileanimation():
    import subprocess
    subprocess.check_call(["/usr/local/bin/ffmpeg","-framerate", "1", "-i", "img/%1d.png", 
                              "-c:v", "libx264", "-r", "30", "-pix_fmt", "yuv420p", "out.mp4"])
    subprocess.check_call(["rm", "-rf",  "img"])

def generateBG(BGpath, verbose=0):
    BG=np.genfromtxt(BGpath,skip_header=1)
    return BG

def average_shots(names,lastnames):
    if names.split()[4]==lastnames.split()[4]:
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

def importdata(directory, verbose=0):
    import os
    if not os.path.exists (directory):
        return -1
    i=0
    lastnames='x y z f 1000'
    xyz_ave=np.zeros([1264,951])
    xyz_old=np.zeros([1264,951])
    j=0
    flag=True
    tx=np.round(np.linspace(0,950*34E-3,num=5))
    txloc=tx/34E-3
    ty=np.round(np.linspace(0,600*34E-3,num=6))
    tyloc=ty/34E-3+400
    try:
        for root, dirs, files in os.walk(directory):
          for names in files:
            if names==".DS_Store" : continue   
            if verbose == 1:
              print('Hashing', names)
            filepath = os.path.join(root,names)
            print(filepath)
            xyz=np.transpose(np.genfromtxt(filepath,skip_header=1))
            flag=average_shots(names,lastnames)
            if flag == True:
                xyz_ave=xyz+xyz_old-BG
                j+=1
            else:
                m=max(xyz_ave.flatten())
                plt.clf()
                plt.imshow((np.clip(xyz_ave/m,0,1)), cmap=plt.cm.viridis, interpolation='nearest')
                lam=float(lastnames.split()[4].split('.')[0])
                plt.title(str(lam)+' nm')
                plt.colorbar()
                plt.ylim([400,1000])
                plt.xticks(txloc,tx)
                plt.yticks(tyloc,ty)
                plt.xlabel("x (mm)")
                plt.ylabel("y (mm)")
                savpath = 'img'
                if not os.path.isdir(savpath):
                    os.makedirs(savpath)  
                plt.savefig(savpath+'/{}.png'.format(i),dpi=250)
                j=0
                i+=1
                xyz_ave=np.zeros([1264,951])
                xyz_old=np.zeros([1264,951])
            lastnames=names
            xyz_old=xyz_ave
        if flag == True:
            xyz_ave=xyz+xyz_old-BG
            j+=1       
        m=max(xyz_ave.flatten())
        plt.clf()
        plt.imshow(np.clip(xyz_ave/m,0,1), cmap=plt.cm.viridis, interpolation='nearest')
        lam=float(lastnames.split()[4].split('.')[0])
        plt.title(str(lam)+' nm')
        plt.colorbar()
        plt.ylim([400,1000])
        plt.xticks(txloc,tx)
        plt.yticks(tyloc,ty)
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        savpath = 'img'
        if not os.path.isdir(savpath):
            os.makedirs(savpath)  
        plt.savefig(savpath+'/{}.png'.format(i),dpi=250)
        os.remove("img/0.png")
        profileanimation()
    except:
        import traceback
        # Print the stack traceback
        traceback.print_exc()
        return -2
  


