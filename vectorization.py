#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 09:48:29 2021

@author: frederiknathan


Module for vectorization of operators 
"""


import sys
from scipy.linalg import *
from scipy import *
import scipy.sparse as sp

from numpy.random import *
from time import *
import os as os 
import os.path as op



def lm(Mat):
    """ construct superoperator corresponding to left multiplication by Mat"""
    D = shape(Mat)[0]
    I = eye(D)
    
    Out = kron(Mat,I)
    return Out
         

def rm(Mat):
    """ construct superoperator corresponding to right multiplication by Mat"""
    D = shape(Mat)[0]
    I = eye(D)
    
    Out = kron(I,Mat.T)

    return Out                   
    
def com(Mat):
    """ Construct superoperator corresponding to commutator with Mat ([Mat,*])"""
    
    return lm(Mat)-rm(Mat)


def mat_to_vec(M):  
    """
    Vectorize matrix
    """
    S = shape(M)
    if len(S)==2:
        
        return ravel(M)
    else:
        NS = S[0]
        return reshape(M,(NS,S[1]*S[2]))
    
def vec_to_mat(V):
    """
    Convert vectorized matrix back to matrix
    """
    D=sqrt(len(V))
    D=int(D+0.1)
    return reshape(V,(D,D))

def get_lindblad(L):
    """
    Construct lindblad superoperator from jump operator
    """
        
    Self_energy= L.conj().T.dot(L)
    
    
    Dissipator = lm(L).dot(rm(L.conj().T))-0.5*(lm(Self_energy)+rm(Self_energy))
    
    return Dissipator
    
def get_trace_vec(dim):
    """ 
    Get vector v corresponding to trace, such that v.dot(X) = trace
    """
    M = eye(dim,dtype=bool)
    return mat_to_vec(M)


# define transpose operator
def get_transpose_operator(DH):
    """
    Generate transpose operator in canonical basis
    """
    
    U = sp.coo_matrix((DH**2,DH**2),dtype=complex)
    U.col = arange(0,DH**2)
    U.row = zeros(DH**2)
    U.data = ones(DH**2)
    for z in range(0,DH**2):
        # find |n><m| corresponding to operator basis state z
        
        n = z//DH
        m = z%DH
        
        # find operator basis state corresponding to |m><n|
        
        z2 = m*DH+n
        U.row[z] = z2
        
    
    U = sp.csc_matrix(U)
    
    return U
def get_hermitian_basis_transformation(DH):
    """
    Generate  operator that transforms from canonical basis to canonical hermitian basis
    """
    
    U = sp.coo_matrix((DH**2,DH**2),dtype=complex)
    U.col = zeros(2*DH**2)
    U.row = zeros(2*DH**2)
    U.data = zeros(2*DH**2,dtype=complex)
    for z in range(0,DH**2):
        # find |n><m| corresponding to operator basis state z
        
        n = z//DH
        m = z%DH
        
        # find operator basis state corresponding to |m><n|
        
        z2 = m*DH+n
        
        
            
        U.col[2*z] = z
        U.col[2*z+1] = z
        
        U.row[2*z] = z
        U.row[2*z+1] = z2
        
        # if n<m, transform to even combination, otherwise odd (if n=m, do nothing)
        if n<m:
            
            U.data[2*z] = sqrt(0.5)
            U.data[2*z+1] = sqrt(0.5)
            
        elif n>m:
            U.data[2*z] = sqrt(0.5)*1j
            U.data[2*z+1] = -sqrt(0.5)*1j
        else:
            U.data[2*z] = 1
            
            
    U = sp.csc_matrix(U)
    U.eliminate_zeros()
    
    return U


    

    
    
    
    
    
    
