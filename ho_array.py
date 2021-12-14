#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 15:46:11 2021

@author: frederiknathan

Module for creating Hamiltonians for arrays of Harmonic oscillators.

All matrices are in the csc format

"""

from numpy import * 
from numpy.linalg import *
import scipy.sparse as sp 
from matplotlib.pyplot import *
def get_bosonic_operators(N):
        
    """Generate bosonic annihilation operator for a harmonic oscillator with the first N photon states included
    """
    
    b = sp.csc_matrix((N,N))
    b[arange(0,N-1),arange(1,N)]=sqrt(arange(1,N))
    
    return b


    
def get_ho_hamiltonian(N,omega):
    """
    Generate Hamiltonian of Harmonic oscillator with first N photon states included and frequency omega. Discrading vacuum energy.
    """
    
    b = get_bosonic_operators(N)
    
    H =  omega * b.T@b
    
    return H


class ho_array():
    """Object representing array of N harmonic oscillators with n_photon states inclued each
    """
    def __init__(self,n_photon):
        assert (type(n_photon)==tuple),"nphoton must be tuple of integers"
        
        self.NO = len(n_photon)
        self.NP = n_photon 

        self.DH = prod(self.NP)
        
        
        I0  = array([sp.eye(n,format="csc") for n in self.NP],dtype=object)
        
        self.b = zeros(self.NO,dtype=object)
        self.nvec = zeros((self.NO,self.DH),dtype=int)
        
        self.A0 = sp.csc_matrix(array([[1]])) 
        self.H  = sp.csc_matrix((self.DH,self.DH))
        self.I  = sp.eye(self.DH,dtype=complex,format="csc")
        for n in range(0,self.NO):
            n1 = prod(self.NP[:n])
            n2 = prod(self.NP[n+1:])
            
            b0  =get_bosonic_operators(self.NP[n])

            self.b[n]=sp.kron(sp.eye(n1,format="csc"),sp.kron(b0,sp.eye(n2,format="csc")),format="csc")
            
            self.nvec[n,:] = arange(0,self.DH)//(self.DH//prod(self.NP[:n+1]))%self.NP[n]
        
    def get_coherent_state(self,alpha):
        """
        Generates direct_product coherent state such that b_n|\psi> = alpha[n]|\psi>
        """
        psi = zeros((self.DH),dtype=complex)
        psi[0] = 1
        for n in range(0,self.NO):
            dpsi = 1*psi
            for m in range(0,self.NP[n]):
                
                dpsi = alpha[n]*(self.b[n].T@dpsi)/(m+1)
                psi = psi+dpsi


        psi = psi/sqrt(sum(abs(psi)**2)) 

        return psi          

    def find_index(self,nlist):
        """
        Find basis vector index of the configuration where oscillator z has np[z] photons (for z  = 0... NO -1)
        np: tuple of ints, dimension NO
        returns int.
        """
        out=0
        
        for z in range(0,self.NO):

           n2 = prod(self.NP[z+1:],dtype=int)
           
           out += n2*nlist[z]
           
        for z in range(0,self.NO):
            assert self.nvec[z][out] == nlist[z],out

        return out
    
    def get_x(self,n):
        return  sqrt(0.5)* ( self.b[n] + self.b[n].T )
    def get_p(self,n):
        return  -1j*sqrt(0.5)* ( self.b[n] - self.b[n].T )
    
    def get_density_matrix(self,nlist):
        """ 
        generate density matrix with photon numbers specified by nlist
        """
        rho = zeros((self.DH,self.DH),dtype=complex)
        

        ind = self.find_index(nlist)

        rho[ind,ind]  = 1 
        
        return rho        
            
    def get_correlation_matrix_from_rho(self,rho):
        """Get correlation matrix from system density matrix rho
        """
        N = len(self.NP)
        k0 = zeros((2*N,2*N),dtype=complex)
        for n1 in range(0,2*N):
            if n1%2 :        
                op1 = self.get_p(n1//2)
            else:
                op1 = self.get_x(n1//2)
            
            for n2 in range(0,2*N):
                if n2%2 :        
                    op2 = self.get_p(n2//2)
                else:
                    op2 = self.get_x(n2//2)
                    
                w = trace(rho@op1@op2)
                
                k0[n1,n2] = w         
                
        return k0

# z0 = (nvec_b==1)* (nvec_s==0)
def gen_3ho_model(NP,Om1,Om2,Om3,C13,C23):
 
    """
    Generate 3-ho-model with 2 HOs coupled to the same HO (HO-3), through the X-. 

    Om1-3 : energies of the 3 HOs
    """
    
    X = ho_array(3,NP)
    
    b1,b2,b3 =[X.b[n] for n in range(0,3)] 
            
    h1 = Om1*b1.T@b1
    h2 = Om2*b2.T@b2
    h3 = Om3*b3.T@b3
    
    h0 = h1+h2+h3
    
    h13 = 0.5 * (b1+b1.T)@(b3+b3.T) * C13
    h23 = 0.5 * (b2+b2.T)@(b3+b3.T) * C23
    
    H = h0 + h13 + h23 
    
    X.H = H
    
    return X




def gen_2ho_nonlinear_model(NP,Om1,Om2,g2,alpha):
    """
    Generate model of 2 harmonic osicllators, coupled nonlinearly
    """
    
    X = ho_array(NP)
    b1,b2 = X.b
    
    h1 = Om1 * b1.T@b1
    h2 = Om2 * b2.T@b2
    C  = g2 * ((b1@b1-X.I*alpha**2)@b2.T+((b1@b1-X.I*alpha**2)@b2.T).conj().T)
    
    H = h1+h2+C
    
    X.H = H
    
    return X


    
    
if __name__=="__main__":
    from units import *
    f1       = 4*GHz
    f2       = 8*GHz
    g2       = 50*MHz
    alpha    = 2 

    
    Om1,Om2  = [2*pi*x for x in (f1,f2)]
    NP       = 10
    
    X=ho_array((30,30))
    # X = gen_2ho_nonlinear_model(NP,Om1,Om2,g2,alpha)
    
    
    out = X.get_coherent_state((3,2))
    
        
        
    M = ho_array((3,7,9))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
        

