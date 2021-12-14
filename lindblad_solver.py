#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:31:38 2021

@author: frederiknathan

Module solving lindblad evolution. So far, only quantum optical master equation (QOME) solver and a 
ULE solver using exact diagonalization of Liouvillian are possible. Add stochastic solvers later
"""

from numpy import *
import ho_array as ho
import spectral_function as sf
import vectorization as vect
from numpy.linalg import *
from matplotlib.pyplot import *
from units import *


def qome_solve(Rholist,Obslist,Llist,H,tlist):
    """
    Solve master equation using QOME. Solve for initial conditions in Rholist, Observables in Obslist, at times in tlist, given
    ULE jump operators in Llist and Hamiltoinian in H (we have no lamb shift)

    Parameters
    ----------
    Rholist        : ndarray(n_rho,D,D). Rholist[n,:,:] gives the nth initial condition for the density matrix to be probed
    Obslist        : ndarray(n_obs,D,D). Obslist[n,:,:] gives the nth observable to be probed
    Llist          : ndarray(n_l,D,D).   Llist[n,:,:]   gives the nth ULE jump operator that is used in the master equation. Each L is split up into D^2 jump operators for the QOME solver
    H              : ndarray(D,D)                       Hamiltonian of the systme
    tlist          : ndarray(NT), float                 Times to be sampled 
    
    Returns
    -------
    
    out            : ndarray(NT,n_obs,n_rho).           Out[nt,no,nr] gives the expectation value of observable Obsli[no] at time tlist[nt] for initial condition Rho(0) = Rholist[nr]


    Objects that enter into master equaton:
           
    
    L_{mn}^k = Llist[k,m,n]
    
    Gamma[n] = sum_{m,k}|L^k_{mn}|^2
    Q[n]     = E[n] - 0.5j Gamma[n]
    R[m,n]   = \sum_k|L^k_{mn}|^2, 
    S[m,n]   = R[m,n] - \delta_{mn}Gamma[n]
    
    We use that 
    
    Tr[X@rho] = X.T.flatten()@rho.flatten()
    
    """
    
    D = shape(H)[0]
    
    for X in [Rholist,Obslist,Llist]:
        assert (shape(X)[-1]==D)*(shape(X)[-2]==D), f"{X.__name__} must be ndarray of dimension ({D},{D}) or (z,{D},{D})"
    
    if len(shape(Rholist))==2:
        Rholist = Rholist.reshape((1,D,D))
    if len(shape(Obslist))==2:
        Obslist = Obslist.reshape((1,D,D))    
    if len(shape(Llist))==2:
        Llist = Llist.reshape((1,D,D))
        
    n_rho = shape(Rholist)[0]
    n_obs = shape(Obslist)[0]
    NT    = len(tlist)
    
    
    ### Transform to eigenbasis of Hamiltonian
    [E,V]  = eigh(H)
    
    rho = V.conj().T@Rholist@V 
    obs = V.conj().T@Obslist @V
    l   = V.conj().T@Llist@V
    
    
    ### Get effective complex energies (Q)
    Gamma = sum(abs(l)**2,axis=(0,1))
    
    Q = E-0.5j*Gamma
    
    
    ### Get matrix Qmat[m,n] = Q[m] - Q[n]^*
    Qmat = outer(Q,ones(D))
    Qmat = Qmat-Qmat.conj().T
    
    
    ### Get rate matrix between eigenstates, R, S: 
    R = sum(abs(l)**2,axis=0)
    S = R - diag(Gamma)
    
    
    ### Diagonalize rate matrix
    [e,v] = eig(S)
    vinv = inv(v)
    
    
    ### Split operators into diagonal and off-diagonal parts    
    
    # Off-diagonal
    obs_od =1*obs
    obs_od[:,arange(D),arange(D)]=0
    
    # Diagonal
    rho_d   = real(rho[:,arange(D),arange(D)])
    obs_d   = real(obs[:,arange(D),arange(D)])
    
    
    ### Transform diagonal components into eigenbasis of S    
    rho_d_eb_0 = 0*rho_d
    obs_d_eb   = 0*obs_d
    
    # Density matrix
    for n in range(0,n_rho):
        
        rho_d_eb_0[n] = vinv@rho_d[n]
        
        
    # Observables. NB! convention is we dont take complex conjugate unless its necessary
    for n in range(0,n_obs):
        
        obs_d_eb[n]  = v.T@obs_d[n]
    
    
    ### Flatten off-diagonal part. Tr[X@rho] = X.T.flatten()@rho.flatten().T
    rho_od_f_0      = rho.reshape((n_rho,D**2))
    obs_od_T_f      = obs_od.swapaxes(1,2).reshape((n_obs,D**2))
    Qmat_f          = Qmat.reshape(D**2)
    
    ### Define output array
    out = zeros((NT,n_obs,n_rho),dtype=complex)
    
    ### Iterate over time
    for nt in arange(0,NT):
        
        # Update time
        t = tlist[nt]
          
        # update off-diagonal part
        rho_od_f = exp(-1j*Qmat_f*t)*rho_od_f_0
        
        # compute expectation value from off-diagonal part
        out_od = obs_od_T_f@rho_od_f.T
    
        # update diagonal part
        rho_d_eb = exp(e*t)*rho_d_eb_0
    
        # compute expectation values from diagonal-part 
        out_d = obs_d_eb@rho_d_eb.T
        
        # store output
        out[nt] = out_d+out_od
            
    
    ### Assert that output is real as a sanity check
    assert amax(abs(imag(out)))<1e-10,"complex expectation values -- did not pass sanity check"
    
    
    return real(out)

    
def solve_with_vectorization(rho0_list,H,L_list,Observable_list,tlist):
    """
    Solve master equation using exact diagonalization of Liouvillian. Solve for initial conditions in Rholist, Observables in Obslist, at times in tlist, given
    ULE jump operators in Llist and Hamiltoinian in H (we have no lamb shift)

    Parameters
    ----------
    Rholist        : ndarray(n_rho,D,D). Rholist[n,:,:] gives the nth initial condition for the density matrix to be probed
    Obslist        : ndarray(n_obs,D,D). Obslist[n,:,:] gives the nth observable to be probed
    Llist          : ndarray(n_l,D,D).   Llist[n,:,:]   gives the nth ULE jump operator that is used in the master equation.
    H              : ndarray(D,D)                       Hamiltonian of the systme
    tlist          : ndarray(NT), float                 Times to be sampled 
    
    Returns
    -------
    
    out            : ndarray(NT,n_obs,n_rho).           Out[nt,no,nr] gives the expectation value of observable Obsli[no] at time tlist[nt] for initial condition Rho(0) = Rholist[nr]

    """
    assert (type(rho0_list)==ndarray) and ndim(rho0_list)==3,"rho0_list must be array of size(N_initalizations,DH,DH)"
    assert (type(L_list)==ndarray) and ndim(L_list)==3,"Lliist must be array of size(N_initalizations,DH,DH)"
    assert (type(Observable_list)==ndarray) and ndim(Observable_list)==3,"Obsevable_list must be array of size(N_observables,DH,DH)"
    
    DH = shape(H)[0]
    
    if DH**2>2000:
        print(f"Warning: hilbert space dimension is large. Exact solver involves diagonalization of {DH**2} x {DH**2} matrix and may take long time")
    if DH**2>25000:
        raise ValueError("Hilbert space dimension is too large for exact solution to be feasible. Use stochastic solver")
    global S_H,S_D,Lv,rho0_vec,obsvec
    S_H = -1j*vect.com(H)
    S_D = 0
    
    
    for L in L_list:
        
        S_D += vect.get_lindblad(L)
    
    Lv = S_H + S_D
    
    rhovec_list = (vect.mat_to_vec(x) for x in rho0_list)
    obsvec_list = (vect.mat_to_vec(x) for x in Observable_list)
    
    rho0_vec = vstack(rhovec_list)
    obsvec   = vstack(obsvec_list)
    
    return solve_liouvillian_with_ed(Lv,tlist,rho0_vec,obsvec)
    

def solve_liouvillian_with_ed(Liouvillian,tlist,rho0vec,obsvec):
    """
    Solve time-evolution with Liouvillian, using exact diagonalization.
    
    Liouvillian:    ndarray, dimension D^2 x D^2. Liouvillian matrix, 
    tlist      :    ndarray, dimension NT. Times to solve for
    rho0_list  :    ndarray, dimension (N,D^2). Vectorized initial states (at tlist[0]). N is the number of initial states to track, D is the original Hilbert space dimension.
    obs_Ø_list :    ndarray, dimension (M,D^2). Vectorized observables to track M is the number of observables
    
    returns 
    
    Out        :    ndarray, dimension (NT,M,N): Expectation value of time-evolved observables. Out[nt,m,n] gives the expectation value of observable m at time tlist[nt], given initialization rho(tlist[0])=rho0_list[n]
    
    """
    
    global rho0_eb,obs_eb,E,V,rho,Out,rho_out
    [E,V]=eig(Liouvillian)
    Vinv = inv(V)
    
    N = shape(rho0vec)[0]
    M = shape(obsvec)[0]
    
    rho0_eb = Vinv@rho0vec.T
    obs_eb  = obsvec@V

    DSH = len(E) # dimension of super hilbert space
    
    E = E.reshape((DSH,1))
    
    # tlist = linspace(0,1e5,10000)
    dt = tlist[1]-tlist[0]
    NT = len(tlist)
    
    Out = zeros((NT,M,N),dtype=float)
    rho_out = zeros((NT,len(rho0_eb)),dtype=complex)
    nt = 0
    # rho = 1*rho0_eb
    for t in tlist:
        
        rho = exp(E*t) * rho0_eb
        Out[nt] = obs_eb@rho
        
            
        nt+=1 
        
        
    return Out

    

    
    
    
    