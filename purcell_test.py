#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 11:33:05 2021

@author: frederiknathan

This script compares the ULE with the exact solution for a gaussian system
that consists of a single oscillator linearly coupled to a finite purcell 
filter of N oscillators, which in turn coupled to an ohmic bath.

The Hamiltonian is given by

H =
      \omega_0 \hat{b}_0^\dagger \hat{b}_0
    + g\hat{x}_0\hat{x_1} 
    + \sum_{n=1}^{N-1} J \hat{x}_n\hat{x}_{n+1}
    + \sum_n \omega \hat b^\dagger_n \hat b_n 
    + \sqrt{\gamma}\hat{x}_N \hat{B}

Here \hat x_n = \sqrt{0.5}(\hat b_n + \hat b^\dagger_n), and \hat B is the bath 
operator of an ohmic bath. 

In the gaussian (exact) solution, the ohmic bath is modelled as a discrete bath 
with N_bath modes. The spectral function has Gaussian cutoff Lambda, 
and is taken to be at temperature temp

We include the first NF oscillators of the filter as a part of the 'system' in 
the ULE simulation, to see if the accuracy improves with NF.
In the ULE simulation we truncate the storage oscillator's Hilbert space at 
NP_s photons, and each incuded filter oscillator at NP_f photons

In this script, the ULE is solved through diagonalization of the Liouvillian.
Since the Liouvllian is a DxD matrix where D = (NP_s**2 * (NP_f)**(2*NF)),
the cost increases very rapidly with NF and NP_s, NP_f.
For better performance for large system, a stochastic solver is probably better.
This has not been coded yet 

"""
 
from basic import tic,toc,I2
from numpy import *
from matplotlib.pyplot import *

import finite_gaussian as gauss
import ho_array as ho
import thermal_bath as tb
import lindblad_solver as lind 
from scipy.linalg import norm 

# =============================================================================
# Parameters
# =============================================================================

# Number of oscillators in filter
N  = 5

# Number of oscillators in filter included as a part of the system in the ULE simulations.
NF = 1

# Number of oscillators in bath
N_bath = 400

# Number of photon states included from storage oscillator in ULE simulations. 
NP_s     = 6

# Number of photon states included per filter oscillator in ULE simulation. 
NP_f   = 4

# Times to be probed 
tlist = arange(0,1000,1)

# Parameters of hamiltonian (i.e. system + filter)
omega0 = 1.2
omega  = 1.5
J      = 0.4
g      = 0.4

# Bath parameters
gamma  = 0.4**2
temp   = 0
Lambda = 3
wmax   = 2

if temp > 1e-15:
    raise ValueError("Exact solution assumes zero temperature. ULE solver must  have temp=0 to model same system. Modify exact solver code if higher temperature desired")


# Parameters that enter in the ULE bath object (only used to compute lamb shift and bath timescales)
w_cutoff = 1*Lambda
dw       = 0.01*omega 

# Spectral function of bath
S = tb.get_ohmic_spectral_function(Lambda,symmetrized=False)

# Frequencies of the discrete gaussian bath
wlist = gauss.generate_bath_spectrum(N_bath,wmax)

# =============================================================================
# Construct gaussian model
# =============================================================================

### Define bath
Bath_g = gauss.discrete_bath(wlist,S)

### Define system

# First get Hamiltonian of filter object (_g means "gaussian" object)
Filter_g = gauss.purcell_filter(N,omega,J)
H_filter = Filter_g.H

# Define system by connecting single oscillator to filter
System_and_filter_g   = gauss.gaussian_system(N+1)

# Set oscillators 1...N to be filter hamiltonian
System_and_filter_g.H[2:,2:] = H_filter

# Set oscillator 0 to be "system"
System_and_filter_g.H[:2,:2] = omega0*0.5*I2

# position coordinates of oscillators 0 and 1
x_0 = System_and_filter_g.get_x(0)
x_1 = System_and_filter_g.get_x(1)

# Define coupling between system and filter
System_and_filter_g.H += 0.5*g*(outer(x_0,x_1)+outer(x_1,x_0))

### Connect system with bath

# Get poisition coordinate of oscillator N (final oscillator in filter)
x_N = System_and_filter_g.get_x(N)

# Generate combined system-bath gaussian object (Combined = system + filter + bath)
Combined         = Bath_g.connect_with_system(System_and_filter_g,sqrt(gamma)*System_and_filter_g.get_x(-1))

# Also get object corresponding to filter and bath (used to get filter spectral function)
Filter_and_bath  = Bath_g.connect_with_system(Filter_g,sqrt(gamma)*Filter_g.get_x(-1))

### Get exact evolution of storage + filter + bath, given initial condition G0 of correlation matrix
def get_G_exact(G0):
    """
    Get exact evolution of expectation values of quadratic combinations of bosonic operators , given initial conditions in G0
    """
    oa = array([System_and_filter_g.get_x(0),System_and_filter_g.get_p(0)]).T
    F = Combined.get_systems_2point_function(G0,operator_array = oa,bath_temperature=temp)
    
    def G_exact(tlist):
        
        out = zeros((len(tlist),2,2),dtype=complex)
        for nt in range(0,len(tlist)):
            out[nt] = F(tlist[nt],tlist[nt])
        
        return out
    
    return G_exact

### Get spectral function of x_1 in the filter + bath system


# Modify shape of spectral function's output
def flatten_output(func):
        
    def out(w):
        if len(shape(w))==1:
            return func(w)[:,0,0]
        elif len(shape(w))==0:
            
            return func(w)[0,0]
        elif len(shape(w))==2:
            return func(w)[:,:,0,0]
        else:
            raise NotImplementedError("Argument dimension not implemented")
            
    return out

# get spectral function
oa = Filter_g.get_x(0).reshape((2*Filter_g.N,1))
SF = flatten_output(Filter_and_bath.get_systems_spectral_function(temp,operator_array=oa,interpolate=True))

# =============================================================================
# Construct ULE model
# =============================================================================

# Divide combined system into the system and bath that enters into the ULE simulation (see docstring in the beginning of script)   
ULE_bath_g    = Combined.truncate(tuple(arange(1+NF,Combined.N))).H
ULE_system_g  = Combined.truncate(tuple(arange(NF+1)))

# Get the part of the filter not included as part of the system (as gaussian object)
ULE_complement_g    = System_and_filter_g.truncate(tuple(arange(1+NF,System_and_filter_g.N)))

# Get the bath seen by the ULE 'system' as a gaussian object (i.e. last (N-NF) filter oscillators + ohmic bath)
ULE_bath_g          = Bath_g.connect_with_system(ULE_complement_g,sqrt(gamma)*ULE_complement_g.get_x(-1))

# Check that we constructed the ULE_bath correctly
assert norm(ULE_bath_g.H - Combined.truncate(tuple(arange(1+NF,Combined.N))).H)<1e-10

### Get spectral function of the bath that enters into ULE simulation
oa = ULE_complement_g.get_x(0).reshape((2*ULE_complement_g.N,1))
SF_ULE = flatten_output(ULE_bath_g.get_systems_spectral_function(temp,operator_array=oa,interpolate=True))

### Get storage ho_array object (i.e., object representing generic array of quantum Harmonic oscillators - not assuming gaussian )

# Number of photon states to include in ULE simulation
NP = (NP_s,) + (NP_f,)*NF

# Get ho_array object (_h stands for 'Hilbert space' or 'Hamiltonian' formulation)
System_h = ULE_system_g.get_ho_array(NP)

# Change Hamiltonian matrix from sparse to dense matrix
System_h.H = System_h.H.toarray()

### Get jump operator

# Get ULE bath object
ULE_Bath = tb.bath(SF_ULE,w_cutoff,dw)

# Get jump operator
if NF ==0 :
    system_bath_coupling = g
elif NF<N:
    system_bath_coupling = J
elif NF==N:
    system_bath_coupling = sqrt(gamma)
else:
    raise ValueError(f"NF must be equal to or smaller than number of filter oscillators ({N})")
    
L = ULE_Bath.get_ule_jump_operator(System_h.get_x(-1).toarray(),System_h.H)*sqrt(2*pi)*system_bath_coupling

# Modify jump operator to have the right dimension to enter in ule solver
Llist = L.reshape((1,)+shape(L))

# Get lamb shift. Uncomment below to include lamb shift (it slows down simulation somewhat)
print("Warning: Lamb shift not included")
# LS =  ULE_Bath.get_ule_lamb_shift_static(System_h.get_x(0).toarray(),System_h.H)*(g)**2
Heff = System_h.H #+ LS

# Define operators (observables) to be meausred during time-evolution
x_h = System_h.get_x(0).toarray()
p_h = System_h.get_p(0).toarray()
obs_list = array([x_h@x_h,p_h@p_h,x_h@p_h])

# Define ULE solver function
def get_ule_solution(rho0):
    rho0list = rho0.reshape((1,)+shape(rho0))
    
    def solver(tlist):
        
        out1 = zeros((len(tlist),2,2),dtype=complex)
        
        out =  lind.solve_with_vectorization(rho0list,Heff,Llist,obs_list,tlist)[:,:,0]
        out1[:,0,0] = out[:,0]
        out1[:,1,1] = out[:,1]
        out1[:,0,1]  = out[:,2]
        out1[:,1,0]  = out[:,2].conj()
        
        return out1
    return solver
# =============================================================================
# Set initial condition
# =============================================================================
### Define initial density matrix as state with 1 photon in storage, and the filter empty. 
rho0=System_h.get_density_matrix((1,)+(0,)*NF)

### Get initial correlation matrix from the initial state

# get initial density matrix of storage
g0    = System_h.get_correlation_matrix_from_rho(rho0)

# get initial density matrix of storage + filter  + bath
G0    = System_and_filter_g.get_vacuum_correlation()
G0[:2*(1+NF),:2*(1+NF)] =  g0

# =============================================================================
# Solve time evolution
# =============================================================================
### get solver functions
s_ule = get_ule_solution(rho0)
s_exact = get_G_exact(G0)

### Solve evolution 
A= s_ule(tlist)
B= s_exact(tlist)
#%%
# =============================================================================
# Plot
# =============================================================================

### Plot evolution of <x^2> for system, using exact solution and ULE
figure(1)
clf()
plot(tlist,real(B[:,0,0]))
plot(tlist,real(A[:,0,0]))
ylim(0,3)
legend(["Exact","ULE"])
titlestr = f"Evolution of photon number in oscillator coupled to ohmic bath through filter of N={N} oscillators.\n"
titlestr+= f"First {NF} filter oscillator(s) included in system in ULE simulation. Osc. Hilbert spaces truncated at {NP} photons"
titlestr += "\n$H = \omega_0 \hat{b}_0^\dagger \hat{b}_0 + g\hat{x}_0\hat{x_1} + \sum_{n=1}^{N-1} J \hat{x}_n\hat{x}_{n+1} + \sum_{n=1}^N\omega \hat b^\dagger_n \hat b_n+\sqrt{\gamma}\hat{x}_N \hat{B}$"
titlestr+=f"\n$\omega_0$={omega0:.2}, g={g:.2}, $\gamma$={gamma:.2}, J={J}, $\omega$={omega:.2},T={temp}"
title(titlestr,fontsize=8)
ylabel("$\\langle x^2\\rangle$")
xlabel("Time")
Gamma = g*ULE_Bath.Gamma0
tau   = ULE_Bath.tau

text(0.99*amax(tlist),2.2,f"$\Gamma = {Gamma:.4}$\n$\\tau = {tau:.4}$",fontsize=9,ha="right",va="top")


### Plot spectral function of bath that enters in the ULE simulation
figure(2)
clf()
plot(ULE_Bath.omrange,ULE_Bath.J(ULE_Bath.omrange))
K0 = amax(ULE_Bath.J(ULE_Bath.omrange))
plot([omega0,omega0],[0,K0*1.1],":k",lw=0.6)
plot([-omega0,-omega0],[0,K0*1.1],":k",lw=0.6)
text(omega0,1.12*K0,"$\omega_0$",ha="center")
text(-omega0,1.12*K0,"$-\omega_0$  ",ha="center")
title(f"Spectral function 'seen' by ULE simulation\n(including first {NF} filter oscillators as a part of system)")
xlabel("Frequency")
ylabel("Value")
ylim((0,1.3*K0))



### Plot spectral function of filter
figure(3)
clf()
plot(ULE_Bath.omrange,SF(ULE_Bath.omrange))
K0 = amax(SF(ULE_Bath.omrange))
plot([omega0,omega0],[0,K0*1.1],":k",lw=0.6)
plot([-omega0,-omega0],[0,K0*1.1],":k",lw=0.6)
text(omega0,1.12*K0,"$\omega_0$",ha="center")
text(-omega0,1.12*K0,"$-\omega_0$  ",ha="center")
title("Spectral function of filter+bath")
xlabel("Frequency")
ylabel("Value")
ylim((0,1.3*K0))






