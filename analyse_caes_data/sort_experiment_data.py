# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 11:49:49 2018

@author: Michael
"""
import shutil
import os
import re
from tqdm import tqdm

def sort_folders(export_folder_name):

    files = os.listdir(export_folder_name)
    made_folders=[]
    
    for names in tqdm(files[0::1]):
        if names==".DS_Store" : continue      
        if names.endswith('.asc') :       
            foldername=export_folder_name+'/'+'_'.join(re.split('_|\.',names)[0:-2])
            
            #print(foldername)
            if os.path.exists(foldername):
                shutil.move(export_folder_name+'/'+names, foldername)
            if not os.path.exists(foldername):
                os.makedirs(foldername)
                shutil.move(export_folder_name+'/'+names, foldername)
                made_folders.append(foldername)
    
    return made_folders