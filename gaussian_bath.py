#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 17:06:30 2021

@author: frederiknathan

Moudule generating spectral functions and jump operators from gaussian baths. 
Core object is Bath.
"""

from matplotlib.pyplot import *
from numpy import *
import numpy.fft as fft
from numpy.linalg import *
import warnings

warnings.filterwarnings('ignore')

def window_function(omega,E1,E2):
    """return window function which is 1 for E1 < omega < E2, and scale of smoothness set by E_softness
    its a gaussian, with center (E1+E2)/2, and width (E1-E2)
    """
    
    Esigma = 0.5*(E1-E2)
    Eav = 0.5*(E1+E2)
    X = exp(-(omega-Eav)**2/(2*Esigma**2))
    if sum(isnan(X))>0:
        raise ValueError
        
    return X
    
def S0_colored(omega,E1,E2,omega0=1):
    """
    Spectral density of colored noise (in our model)
    """
    A1 = window_function(omega, E1, E2)
    A2 = window_function(-omega, E1, E2)
    
    return (A1+A2)*abs(omega)/omega0

def get_ohmic_spectral_function(Lambda,omega0=1,symmetrized=True):
    """
    generate spectral function

    S(\omega) = |\omega| * e^{-\omega^2/2\Lambda^2}/\omega_0
    
    Parameters
    ----------
    Lambda : float
        Cutoff frequency.
    omega0 : float, optional
        Normalization. The default is 1.
    symmetrized : bool, optional
        indicate if the spectral function shoud be symmetric or antisymmetric. If False, |\omega| -> \omega in the definition of S. 
        The default is True.

    Returns
    -------
    S : method
        spectral function S,

    """
    if symmetrized :
            
        def f(omega):
            return abs(omega)*exp(-(omega**2)/(2*Lambda**2))/omega0
    
    else:
        def f(omega):
            return (omega)*exp(-(omega**2)/(2*Lambda**2))/omega0
        
    return f

def S0_ohmic(omega,Lambda,omega0=1):
    """
    Spectral density of ohmic bath
    
        S(omega) = |omega|*e^(-omega^2/(2*Lambda**2))
         
    """
    
    Out = abs(omega)*exp(-(omega**2)/(2*Lambda**2))/omega0
    
    
    return Out
def BE(omega,Temp=1):
    """
    Return bose-einstein distribution function at temperature temp. 
    """
    return (1)/(1-exp(-omega/Temp))*sign(omega)

def get_J_colored(E1,E2,Temp,omega0=1):
    """
    generate spectral function of colored bath at given values of E0,E1,Temp 
    
    Returns spectral function as a function/method
    """
    dw = 1e-12
    nan_value = S0_colored(dw,E1,E2,omega0=omega0)*BE(dw,Temp=Temp)
    def J(omega):
        
        return nan_to_num(S0_colored(omega,E1,E2,omega0=omega0)*BE(omega,Temp=Temp),nan=nan_value)
    
    return J

def get_J_ohmic(Temp,Lambda,omega0=1):
    """
    generate spectral function of ohmic bath, modified with gausian as cutoff,
    
    S(omega) = |omega|*e^(-omega^2/(2*Lambda**2))

    """
    def J(omega):
        return nan_to_num(S0_ohmic(omega,Lambda,omega0=omega0)*BE(omega,Temp=Temp),nan=Temp/omega0)
    
    
    return J 

def get_J_from_S(S,temperature,zv):
    """
    Get bath spectral function from bare spectral function at a given temperature. Zv specifies what value to give at zero (Where BE diverges)"""
    

    def out(energy):
        return nan_to_num(BE(energy,Temp = temperature)*S(energy)*sign(energy),nan=zv)
    
    return out

def get_g(J):
    """
    Get jump spectral function from given bath spectral function, J
    input:method
    output: method
    """
    
    def g(omega):
        return sqrt(J(omega)/(2*pi))
    
    return g


def get_ft_vector(f,cutoff,dw):
    """
    Return fourier transform of function as vector, with frequency cutoff <cutoff> and frequency resolution <dw>
    Fourier transform, \int dw e^{-iwt} J(w)
    """
    
    omrange = linspace(-cutoff,cutoff,2*int(cutoff/dw)+1)[:-1]
    n_om  = len(omrange)
    omrange = concatenate((omrange[n_om//2:],omrange[:n_om//2]))
    
    vec    = fft.fft(f(omrange))*dw 
    # Jvec    = (-1)**(arange(0,n_om))*Jvec
    times   = 2*pi*fft.fftfreq(n_om,d=dw)
    AS = argsort(times)
    times = times[AS]
    vec = vec[AS]
    
    return times,vec
        
        
        
class bath():
    """
    bath object. Takes as input a spectral function. Computes jump correlator 
    and ULE timescales automatically. 
    Can plot correlation functions and spectral functions as well as generate 
    jump operators and Lamb shfit
    
    Parameters
    ----------
        J : callable.     
            Spectral function of bath. Must be real-valued
        cutoff : float, >0.    
            Cutoff frequency used to compute time-domain functions (used to 
            compute Lamb shift and ULE timescales, and for plotting correlation 
            functions).
        dw : float, >0.  
            Frequency resolution to compute time-domain functions (see above)
        
    Properties
    ----------
        J : callable.  
            Spectral function of bath. Same as input variable J
        g : callable.  
            Fourier transform of jump correlator (sqrt of spectral function)
        cutoff : float.    
            Same as input variable cutoff
        dw : float.     
            Same as input variable dw
        dt : floeat.    
            Time resoution in time-domain functions. Given by pi/cutoff
        omrange : ndarray(NW)    
            Frequency array used as input for computation of time-domain 
            observables (see above). Frequencies are in range (-cutoff,cutoff)
            and evenly spaced by dw. Here NW is the length of the resulting 
            array.
        times : ndarray(NW)     
            times corresponding to time-domain functions
        correlation_function : ndarray(NW), complex
            Correlation function at times specified in times. 
            Defined such that correlation_function[z] = J(times[z]).
        jump_correlator  :ndarray(NW), complex    
            Jump correlator at times specified in times 
        Gamma0 : float, positive.    
            'bare' Gamma energy scale. The ULE Gamma energy scale is given by 
            gamma*||X||*Gamma0, where gamma and ||X|| are properties of the 
            system-bath coupling (see ULE paper), and not the bath itself. 
            I.e. gamma, ||X|| along with Gamma0 can be used to compute Gamma.
        tau : float, positive.      
            Correlation time of the bath, as defined in the ULE paper.
        

    """
    
    def __init__(self,J,cutoff,dw):
        self.J = J
        self.g = get_g(J)
        
        self.cutoff = cutoff
        self.dw     = dw
        self.dt     = 2*pi/(2*cutoff)
        
        # Range of frequencies 
        self.omrange = linspace(-cutoff,cutoff,2*int(cutoff/dw)+1)[:-1]

        self.times,self.correlation_function = self.get_ft_vector(self.J)
        Null,self.jump_correlator = self.get_ft_vector(self.g)
                
        
        g_int = sum(abs(self.jump_correlator))*self.dt 
        
        self.Gamma0 = 4*g_int**2
        
        self.tau = sum(abs(self.jump_correlator*self.times))*self.dt/g_int 
        
    def plot_jump_correlator(self,nfig=1):
        """Plot jump correlator as a function of time, evaluated at times in self.times.
        """
        
        figure(nfig)
        
        plot(self.times,abs(self.jump_correlator))
        title(f"Jump correlator (abs)")
        xlabel("Time")
        
        
    def plot_spectral_function(self,nfig=2):
        """
        Plot spectral function, evaluated at frequencies in self.omrange.
        """
        figure(nfig)
 
        plot(self.omrange,self.J(self.omrange))
        title(f"Spectral function")
        xlabel("$\omega$")


    def get_ft_vector(self,f) :
        """
        Return fourier transform of function as vector, with frequency cutoff <cutoff> and frequency resolution <dw>
        Fourier transform, \int dw e^{-iwt} J(w)
        """
        cutoff= self.cutoff
        dw    = self.dw
        omrange = linspace(-cutoff,cutoff,2*int(cutoff/dw)+1)[:-1]
        n_om  = len(omrange)
        omrange = concatenate((omrange[n_om//2:],omrange[:n_om//2]))
        
        vec    = fft.fft(f(omrange))*dw 
        # Jvec    = (-1)**(arange(0,n_om))*Jvec
        times   = 2*pi*fft.fftfreq(n_om,d=dw)
        AS = argsort(times)
        times = times[AS]
        vec = vec[AS]
        
        return times,vec
    
    def get_ule_jump_operator(self,X,H):
        """
        Get jump operator for bath, associated with operator X and Hamiltonian H
        (all must be arrays)
        """

                
        [E,V]=eigh(H)
        ND = len(E)
        Emat = outer(E,ones(ND))
        Emat = Emat.T-Emat
         
        X_eb = V.conj().T @ X @ V
        
        # print(self.g(Emat))
        L = sqrt(2*pi)*V@(X_eb * self.g(Emat))@(V.conj().T)
        L = L * (abs(L)>1e-13)
        
        return L
        
        
        
    def get_cpv(self,f,real_valued=True):
        """ 
        Return Cauchy principal value of integral \int dw f(w)/w 
        
        The integral is defined as
        
        Re ( \int dw f(w)Re ( 1/(w-i0^+)))
        
        This is the same as 
        
        i/2 *  \int_-\infty^\infty dt f(t)e^{-0^+ |t|} sgn(t)
            
        where  f(t) =     \int d\omega f(\omega)e^{-i\omega t} 
        
        (i.e. get_time_domain_function(f))  
        
        """
        
        
    
        # times,F = get_time_domain_function(g,freqvec) 
        Null,F = self.get_ft_vector(f)
        
        # self.F = F
        
        N = len(F)
        
        F[N//2:]=-F[N//2:]
        F[0]=0
        # dt = tvec[1]-tvec[0]
        S =-0.5j*sum(F)*self.dt
        
        if real_valued:
            S=real(S)
        return S
                
            
    def get_lamb_shift_amplitude(self,q1,q2):
        """
        Get amplitude of lamb shift F_{\alpha \beta }(q1,q2) (see L)
        """
        def f(x):
            return self.g(x-q1)*self.g(x+q2)
    
        return -2*pi*self.get_cpv(f)        
    def get_ule_lamb_shift_static(self,X,H):
        """Get ULE Lamb shift for a static Hamiltonian, using self.get_ft_vector to calculate cauchy p.v.
        
        The cpv calculation can definitely be parallelized for speedup"""
             
        [E,U]=eigh(H) 
        D  = shape(H)[0]
    
        X_b = U.conj().T.dot(X).dot(U)
     
        LS_b = zeros((D,D),dtype=complex)
        
        
        # g = get_jump_correlator(J)
        print("Computing Lamb shift")
        for m in range(0,D):
            if m%(max(1,D//10))==0:
                print(f"    At column {m}/{D}")
            for n in range(0,D):

                for l in range(0,D):
                    E_mn = E[m]-E[n]
                    E_nl = E[n]-E[l]
                    
                    Amplitude = self.get_lamb_shift_amplitude(E_mn,E_nl)
                    
                    
                    LS_b[m,l]+=Amplitude*X_b[m,n]*X_b[n,l]
        
        LS = U.dot(LS_b).dot(U.conj().T)
       
        
        return LS 

 
        
        
        
        





if __name__ == "__main__":
    
    ### Testing and plotting
    Es = 0.4
    E1 = 5.5
    E2 = 10.5
    Temp = 0.1
    cutoff = 50
    dw = 0.001
        
    omrange = linspace(-50*E1,50*E1,10000)
    
    J = get_J_ohmic(1,10)
    
    
    
    B = bath(J,cutoff,dw)
    B.plot_jump_correlator(nfig=1)
    xlim(-5,5)
    B.plot_spectral_function(nfig=2)
    print(f"Gamma/gamma = {B.Gamma0:.4}")
    print(f"tau         = {B.tau:.4}")









