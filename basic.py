#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:17:30 2019

@author: frederik
# """
# from numpy import *
# from scipy import *
# from scipy.integrate import *
# from scipy.linalg import *
import time
# from numpy.fft import * 
import numpy as np
import os 
import sys

import datetime as datetime 

GHz = 1 
MHz = 0.001
THz = 1000
ns  = 1 
mus = 1/MHz 
ps  = 1/THz 
K   = 20.837
Tesla  = 1

SZ = np.array([[1,0],[0,-1]])*(1+0j)
SX = np.array([[0,1],[1,0]])*(1+0j)
SY = np.array([[0,-1j],[1j,0]])
SM = np.array([[0,0],[1,0]])
SP = np.array([[0,1],[0,0]])
I2 = np.array([[1,0],[0,1]])*(1+0j)

#class Basic():
#    def __init__(self):
#        
old_stdout = sys.stdout
old_stderr = sys.stderr
#
#BASIC= Basic()

def Redirect_output(File):  
    
    global old_stdout
    global old_stderr
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    sys.stdout = open(File, 'a')
    sys.stderr = open(File, 'a')
        
    
def Reset_std():
#    sys.stdout.close()
    
    
    global sys 
    
    sys.stdout = old_stdout 
    sys.stderr = old_stderr
#    sys.stderr=sys.__stderr__
    
def get_pauli_components(Mat,real_output=False):
    v0 = 0.5*trace(Mat)
    vx = 0.5*trace(SX.dot(Mat))
    vy = 0.5*trace(SY.dot(Mat))
    vz = 0.5*trace(SZ.dot(Mat))
    
    Out = array([v0,vx,vy,vz])
    
    if real_output:
        Out = real(Out)
    return Out

def signstr(value):
    if value>=0:
        return "+"
    if value < 0:
        return "-"
def get_bloch_vector(Psi):

    Psi = Psi/norm(Psi)

    Rho = outer(Psi,Psi.conj().T)

    S=2*get_pauli_components(Rho,real_output=1)


    return S[1:]
            
def TimeStamp():
    return datetime.datetime.now().strftime("%d/%m %H:%M")
    
    
def ID_gen():
    timestring=datetime.datetime.now().strftime("%y%m%d_%H%M-%S.%f")[:-3]
    
    return timestring 

def expval(psi,A,hermitian=True):
    """
    Expectation value
    """
    
    global X 
    
    X = sum(psi.conj()*(A.dot(psi)),axis=0)
    
    if hermitian:
        
        return real(X)
    
    else:
        return X
    
    
def RecursiveSearch(Pattern,Dir,nrec=0,criterion=""):
    FileList=os.listdir(Dir)
    PathList=[]
    PathFound=False
    
    for x in FileList:
        Path=Dir+"/"+x



        if os.path.isdir(Path):
            A=RecursiveSearch(Pattern,Path,nrec=nrec+1)
            PathList=PathList+A
                
            
        elif PatternMatch(Pattern,x):
            if criterion in Path:
                
                PathList.append(Path)
    
    
    if len(PathList)==0:
        if nrec>0:
            return []
        else: 
            raise FileNotFoundError("No files matched the search criterion \"%s\" "%Pattern)
    elif len(PathList)>1:
        raise FileNotFoundError("Multiple files match the pattern %s . Be more specific"%Pattern)
    
    else:
        if nrec>0:
            return PathList
        else:
            return PathList[0]


    
def FileNamePatternMatch(Pattern,Directory):
    FileList=os.listdir(Directory)
    
    nmatch=0
    OutFileList=[]
    for n in range(0,len( FileList)):
        file=FileList[n]
        K=PatternMatch(Pattern,file)
        

        
        if K:
            Kre=PatternMatch("_re.npz",file)
            if not Kre:
                nmatch+=1
                OutFileList.append(file)
            
    
    if nmatch==1:
        return OutFileList[0]
    elif nmatch==0:
        raise FileNotFoundError("No files found that matched the pattern %s"%Pattern)
    else:
        print(nmatch)
        print(OutFileList)
        raise FileNotFoundError("%d files match the pattern %s: %s. Be more specific"%(nmatch,Pattern,str(OutFileList)))
        
            

def PatternMatch(Pattern,Name):
    Lp=len(Pattern)
    Ln=len(Name)
    for offset in range(0,Ln-Lp+1):
        TestStr=Name[offset:offset+Lp]
        if TestStr==Pattern:
            return 1
            
    
    return 0    

def get_ct():
    return time.time()

global T_tic 
T_tic= get_ct()
tic_list = np.ones(500)*get_ct()
def tic(n=None):
    global T_tic
    global tic_list
    
    if n==None:
        
        T_tic = get_ct()
    else:
        tic_list[n]=get_ct()
    
def toc(n=None):
    if n == None:
        return get_ct()-T_tic
    else:
        return get_ct()-tic_list[n]
    
    
