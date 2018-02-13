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

xlimit=145
ylimit=145
pixpermm=31.32*1000

def makefitsize(biases,stdy, color='b',label='None'):
    fitsize=so.curve_fit(sizefit,biases,np.array(stdy)/pixpermm,p0=0.01,bounds=(0,1))
    plt.gcf()
    plt.plot(biases,sizefit(biases,fitsize[0]),color=color)
    plt.xlabel("Einzel lens bias / V")
    plt.ylabel("Electron bunch size / mm")
    
    return fitsize;
    
def makefitT(biases,stdy, color='b',label='None'):
    fitT=so.curve_fit(tempfit,biases,np.array(stdy)/pixpermm,p0=[5,3,0],bounds=(-50,50))
    plt.gcf()
    #plt.plot(biases,np.array(stdy)/pixpermm,color=color,marker='.',linestyle='None',label=label+', {:} K, '.format(fitT[0][0]))#+ '{:.1f} $\mu$m, '.format(fitT[0][1]*1E6))
    fittedfunction=tempfit(biases,fitT[0][0],fitT[0][1],fitT[0][2])#,fitT[0][2])
    plt.plot(biases,fittedfunction,color=color,label=label+', {:.1f} K, '.format(fitT[0][0]*10)+ '{:.1f} $\mu$m, '.format(fitT[0][1]*1E2))
    plt.xlabel("Einzel lens bias / V")
    plt.ylabel("Electron bunch size / mm")
    #print(biases[np.argmin(fittedfunction)])
    #print(biases[np.argmin(stdy)])
    
    return fitT;

A_tab=np.genfromtxt('../../experimental_data/15-11-17/A_and_B/A_vs_z_vs_BE.txt')
B_tab=np.genfromtxt('../../experimental_data/15-11-17/A_and_B/B_vs_z_vs_BE.txt')
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
    N=500
    #sigmaz=3E-3
    for blur_z in np.random.normal(loc=0, scale=abs(sigmaz), size=N):
        Ablur=A(E,z+blur_z)+Ablur
 #   Ablur=A(E,z)   
    return Ablur/N
    
def B_blur(E,z,sigmaz):
    Bblur=0
    N=500
    #sigmaz=3E-3
    for blur_z in np.random.normal(loc=0, scale=abs(sigmaz), size=N):
        Bblur=B(E,z+blur_z)+Bblur
#    Bblur=B(E,z)
    return Bblur/N

def tempfit(U, T, sigmay, z):#, yfwhm):
    U=np.array(U)    

    E=1500
    T=T*10
    #sigmay=sigmay*1E-4
    #z=-3E-3
    y=(np.sqrt(abs((A(U,z)**2*sigmay**2)+B(U,z)**2*(sc.k*T)/(2*sc.e*E)))) #0.506211 is conversion factor between
                                                                            #the voltage on the repel plate and the beam energy
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


def find_axis(xyz,xo,yo,threshold):
    xa=np.array([-1000,1000])
    m_arr=[]
    l_arr=[]
    #boundingbox=np.where(xyz > threshold)
    #plt.scatter(boundingbox[1],boundingbox[0],color='C3')
    for theta in np.arange(-sc.pi,sc.pi,0.1):
        ya=np.tan(theta)*(xa-xo)+yo
        length = int(np.hypot(xa[1]-xa[0], ya[1]-ya[0]))
        if length < 100000:
            xt, yt = np.linspace(xa[0], xa[1], length), np.linspace(ya[0], ya[1], length)
            delete=np.array([*np.where(xt<0)[0], *np.where(xt>=len(xyz[0]))[0], *np.where(yt<0)[0], *np.where(yt>=len(xyz)-2)[0]])
            yt=np.delete(yt,delete.astype(np.int))
            xt=np.delete(xt,delete.astype(np.int))
            l_t = sum(np.where(xyz[yt.astype(np.int), xt.astype(np.int)]> threshold, 1, 0))
            l_arr.append(l_t)
            m_arr.append(np.tan(theta))
            
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
    
    l_minor=sum(np.where(xyz[yt.astype(np.int), xt.astype(np.int)]> threshold, 1, 0))
#    print('minor and major:')
#    print(l_minor)
#    print(l_major)
    
    return l_major/2.634, l_minor/2.634, sc.pi-np.arctan(m_at_max_l)
    

def initial_guess(xyz,threshold_perc=99,flag=False):
    xyz_blurred=gaussian_filter(xyz, sigma=len(xyz[0])/50)
    amplitude=max(xyz_blurred.flatten())
    threshold=np.percentile(xyz_blurred.flatten(),threshold_perc)
    if flag==True:
        threshold=amplitude*0.75
    yo,xo=scim.center_of_mass(np.where(xyz_blurred > threshold, 1, 0))
    
    #plt.imshow(xyz)

    #plt.plot(xo,yo,marker='o')
    
    #plt.show()
    
    #plt.clf()
    
    sigma_x,sigma_y,theta=find_axis(xyz_blurred,int(np.round(xo)),int(np.round(yo)),threshold)
    offset=np.mean(xyz[:10,:10].flatten())
    #print(amplitude, xo, yo, sigma_x, sigma_y, theta, offset)
    
    return (amplitude, xo, yo, sigma_x, sigma_y, theta, offset)
    
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

def profileanimation(outfile):
    import subprocess
    subprocess.check_call(["/usr/local/bin/ffmpeg","-framerate", "30", "-i", "img/%1d.png", 
                              "-c:v", "libx264", "-r", "30", "-pix_fmt", "yuv420p", outfile])
    subprocess.check_call(["rm", "-rf",  "img"])

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
         
def align_shots(z,ig):
    y0=ig[1]#len(z)/2#
    x0=ig[2]#len(z[0])/2#
    zx=len(z[0])
    zy=len(z)
    xlimit=int(3*ig[3])
    ylimit=int(3*ig[3])
    #print([int(x0)-xlimit,int(x0)+xlimit,int(y0)-ylimit,int(y0)+ylimit])
    return z[np.clip(int(x0)-xlimit,0,zx):np.clip(int(x0)+xlimit,0,zx),np.clip(int(y0)-ylimit,0,zy):np.clip(int(y0)+ylimit,0,zy)]
    

def importdata(directory, verbose=0, biases=[], outdir='out', outfile='none_given',ptype='e'):
    import os
    if not os.path.exists (directory):
        return -1
    i=0
    
    if outfile == 'none_given':
        outfile = directory.split('/')[-1] 
        outdir = directory.split('/')[-3]+'/'+directory.split('/')[-2]
    
    expt_name=outdir+outfile
    j=0
    stdx=[]
    stdy=[] 
    x0=[]
    y0=[]
    estdx=[]
    estdy=[]
    biases=[]
    jlist=[]
    ig_stdx=[]
    ig_stdy=[]
    sumpix=[]
    theta=[]
    l=0
    k=0
    
    try:
        for root, dirs, files in os.walk(directory):
          for names in tqdm(files,smoothing=0):
            if names==".DS_Store" : continue   
                    
            if verbose == 1:
              print('Hashing', names)
            filepath = os.path.join(root,names)
            
            xyz=np.transpose(np.genfromtxt(filepath,skip_header=1))
            header=np.genfromtxt(filepath, max_rows=1)
            
            #only want to try and fit when there is a spot to fit to. 
            
            std=np.std(xyz)
            
            if std<1: 
#                plt.imshow(xyz)
#                plt.title(std)
#                plt.show()
#                plt.clf()
                continue
            
            #flag=False#average_shots(names,lastnames,header,lastheader)
            ig=initial_guess(xyz)
            plt.clf()
            z_crop=align_shots(xyz,ig)
            ig_ave=initial_guess(z_crop,flag=True)  
            x = np.linspace(0,len(z_crop),len(z_crop))
            y = np.linspace(0,len(z_crop[0]),len(z_crop[0]))
            x,y = np.meshgrid(x, y)
            xy=x,y
#            plt.clf()
#            plt.imshow((np.clip(z_crop/ig_ave[0],0,1)), cmap=plt.cm.viridis, interpolation='nearest')
#            plt.show()
            #lam=float(names.split()[4].split('.')[0])
            if ptype=='i':
                bias=header[4] 
            
            else:bias=header[1]
            
            try:                              
                popt, pcov = so.curve_fit(twoD_Gaussian, xy, z_crop.reshape(len(z_crop)*len(z_crop[0])), p0=ig_ave, bounds=bounds(ig_ave))
                stdx.append(popt[3]) 
                stdy.append(popt[4])
                x0.append(popt[1])#+ig[1]-ig[3])
                y0.append(popt[2])#+ig[2]-ig[3])
                stderr=np.diag(pcov)
                estdx.append(stderr[3]) 
                estdy.append(stderr[4])
                biases.append(bias)
                ig_stdx.append(ig_ave[1])
                ig_stdy.append(ig_ave[2])
                sumpix.append(sum(sum(z_crop)))
                jlist.append(j)
                theta.append(popt[5])
                
                if not os.path.exists('img'):
                    os.makedirs('img')
                 
                l=l+1
                
                plt.imshow(z_crop, cmap=plt.cm.viridis, interpolation='nearest')
                plt.colorbar()
                plt.xlabel("x (mm)")
                plt.ylabel("y (mm)")
                plt.title(bias)
                Z=twoD_Gaussian_mesh(xy,*ig_ave)
                Z1=twoD_Gaussian_mesh(xy,*popt)
                plt.contour(x, y, Z,linewidths=0.5,colors='C0')
                plt.contour(x, y, Z1,linewidths=0.5,colors='C1')
                plt.savefig('img/{}.png'.format(k),dpi=250)
                plt.clf()
                k+=1
                l=0
#                

                
            except:
                import traceback
                # Print the stack traceback
                traceback.print_exc()
                print(names)
                print(bias)
                print(ig_ave)
                print(bounds(ig))
                
                
            
            j=1
            i+=1
        
#        print(expt_name.split('/')[-1]+".mp4")
        if os.path.exists(expt_name.split('/')[-1]+".mp4"):
            os.remove(expt_name.split('/')[-1]+".mp4")
        profileanimation(expt_name.split('/')[-1]+".mp4")
        plt.clf()
        
        #print(makefitT(biases,np.array(stdy)))
        
        #plt.savefig('out/'+expt_name+"biases_vs_stdx.svg")
        
        #Save the width and position of the fitted gaussains in terms of pixels
       
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            
        np.savetxt(outdir+'/'+outfile+'.txt', np.transpose([biases,stdx,stdy,estdx,estdy,x0,y0,theta]),header='biases stdx stdy errstdx errstdy x0 y0 theta' )
        
        return biases,stdx,stdy,estdx,estdy
    except:
        import traceback
        # Print the stack traceback
        traceback.print_exc()
        print(names)
        print(bias)
        print(ig)
        print(bounds(ig))
        plt.imshow(xyz)
        

  
def importfolder(directory,ptype='e'):
    import os
    if not os.path.exists (directory):
        return -1
    
    for root, dirs, files in os.walk(directory):
        for folders in dirs:
            importdata(directory+'/'+folders,ptype=ptype)
            print(folders+' recorded')
            
def strip_outliers(data):
    outliers=np.where(data['errstdy']>0.02)#data['stdy']/100)
    return np.delete(data,outliers)

def strip_low_repeats(stat,count):
    outliers=np.where(count[0]<20)
    return np.delete(stat,outliers)
            
def plot_waist_scan(directory):
    import os
    import operator
    
    if not os.path.exists (directory):
        return -1
    i=0
    w=[]
    for root, dirs, files in os.walk(directory):
        for names in files:
            print(names)
            if names==".DS_Store" : continue
            if "vs" in names : continue
            wl=float(names.split('.')[0])
            if wl > 1000: wl=wl/10
            w.append(wl)
        wlrange=max(w)-min(w)
        print(wlrange)
        
        for names in files:
            if names==".DS_Store" : continue
            if "vs" in names : continue
            #extract waist scans from files
            data=np.genfromtxt(directory+'/'+names, names=True, delimiter=' ')
            #print(data)
            #extract wavelength from file name
            wavelength=float(names.split('.')[0])
            if wavelength > 1000: wavelength=wavelength/10
            
            if wlrange>0:
                cx=plt.cm.viridis((wavelength-min(w))/wlrange)
                cy=plt.cm.viridis((wavelength-min(w))/wlrange)
            else : 
                cx='C0' 
                cy='C1'
            #data=strip_outliers(data)
            
            #print(data)
            
            #plt.plot(data['biases'], data['stdx'],'x')
            
            unique, counts = np.unique(data['biases'], return_counts=True)
            
            if np.mean(counts)>2:
                grouped_mean_x=ss.binned_statistic(data['biases'], data['stdx'], statistic='mean', bins=np.unique(data['biases']))
                group_error_x=ss.binned_statistic(data['biases'], data['stdx'], statistic=np.std, bins=np.unique(data['biases']))
                group_count_x=ss.binned_statistic(data['biases'], data['stdx'], statistic='count', bins=np.unique(data['biases']))
    
                grouped_mean_y=ss.binned_statistic(data['biases'], data['stdy'], statistic='mean', bins=np.unique(data['biases']))
                group_error_y=ss.binned_statistic(data['biases'], data['stdy'], statistic=np.std, bins=np.unique(data['biases']))
                group_count_y=ss.binned_statistic(data['biases'], data['stdy'], statistic='count', bins=np.unique(data['biases']))
                
                grouped_mean_strip_x=strip_low_repeats(grouped_mean_x[0], group_count_x)
                group_error_strip_x=strip_low_repeats(group_error_x[0], group_count_x)
                biases_strip_x=strip_low_repeats(group_error_x[1][:-1], group_count_x)
                
                grouped_mean_strip_y=strip_low_repeats(grouped_mean_y[0], group_count_y)
                group_error_strip_y=strip_low_repeats(group_error_y[0], group_count_y)
                biases_strip_y=strip_low_repeats(group_error_y[1][:-1], group_count_y)
                
            else:
                n=0
                grouped_mean_strip_x=data['stdx'][n:]
                group_error_strip_x=data['errstdx'][n:]
                biases_strip_x=data['biases'][n:]
                
                grouped_mean_strip_y=data['stdy'][n:]
                group_error_strip_y=data['errstdy'][n:]
                biases_strip_y=data['biases'][n:]
            
            #print(group_error[0])
            #f=plt.plot(group_error_y[1][:-1], group_count_y[0], marker='.',linestyle='None',color=cx)
            
            plt.figure(3)
            
            if len(grouped_mean_strip_y)>10:      
                #f=plt.errorbar(biases_strip_x, grouped_mean_strip_x/pixpermm, yerr=group_error_strip_x/pixpermm, marker='.',linestyle='None',color=cx)
                print(makefitT(biases_strip_y,grouped_mean_strip_y,color=cy,label='')[0])
                k=plt.errorbar(biases_strip_y, grouped_mean_strip_y/pixpermm, yerr=group_error_strip_y/pixpermm, marker='.',linestyle='None',color=cy)
                #print(makefitT(biases_strip_x,grouped_mean_strip_x,color=cx,label=str(wavelength)+' uW x')[0])

            #print(min(data['stdx']/pixpermm))
            
        #handles, labels = plt.gca().get_legend_handles_labels()
        #hl = sorted(zip(handles, labels),key=operator.itemgetter(1))
        #handles2, labels2 = zip(*hl)
        #plt.gca().legend(handles2, labels2)
        
        plt.gca()
        plt.legend()
        
        plt.xlabel('Einzel lens bias (V)')
        plt.ylabel('Beam size at MCP (m)')
        #plt.ylim([0.13,0.25])
        #plt.xlim([200,600])
        
    plt.savefig('out/waist_scans.svg')
    
            

