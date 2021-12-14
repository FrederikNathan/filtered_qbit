#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 08:11:18 2021

@author: frederiknathan

Mode for solving dynamics of finite quadratic systems of bosons

Key object is the gaussian_system class which contains properties and useful methods for solving a gaussian system. 
The code also works for the case where the array of oscillators system can be divided into a "system" and "bath", 
where the bath has a given spectral function and temperature. The bath is discrete still.



"""

from basic import tic,toc
from numpy import *
from numpy.linalg import *
from scipy.linalg import * 
import numpy.random as npr
import spectral_function as sf
from matplotlib.pyplot import *
import numpy.fft as fft 
import ho_array as ho
from scipy.interpolate import interp1d 



def generate_bath_spectrum(Nbath,wmax):
    """
    Generate uniformly spaced frequency spectrum of bath

    Parameters
    ----------
    Nbath : int
        Number of modes in the bath.
    wmax : float, positive
        maximal frequency in the spectrum.

    Returns
    -------
    wlist : ndarray(Nbath), float
        wlist[n] = (n+1)*dw, where dw = wmax/Nbath

    """
    assert type(Nbath)==int and Nbath>0,"Nbath must be postive integer"
    assert wmax>0,"wmax must be positive float"
    

    return arange(1,Nbath+1)*wmax/Nbath

def bose_einstein(energy,temperature):
    """
    Give Bose einstein distribution at a given temperature

    Parameters
    ----------
    energy       : ndarray. Array of energies to be probed
    temperature  : float. Temperature of bath

    Returns
    -------
    weights      : ndarray. Bose-einstein values at the energies probed.


    """
    return 1/(exp(energy/temperature)-1)

def get_symplectic_matrix(N):
    """
    Construct symplectic matrix for N degrees of freedom.
    """
    
    C = zeros((2*N,2*N),dtype=complex)
    
    for n in range(0,N):
        C[2*n,2*n+1] = 1j
        C[2*n+1,2*n] = -1j
    
    return C

class gaussian_system():
    """
    Object of quadratic system of bosons, with Hamitlonian
    
    H = \sum_{mn} A_m A_n H_{mn}
    
    where [A_m,A_n] = C_{mn}, with C the symplectic matrix (see above)
    
    Parameters:
        
    N: int. Number of modes in the system
    H: ndarray of floats shape(2N,2N): Hamiltonian matrix of system. Should be positive definite
    
    Properties
    H: Hamiltonian (See above)
    C: symplectic matrix of the system
    E: Eigenvalues of HC
    V: Eigenvectors of HC
    Vinv: inverse of V.
    diagonalized, bool. Flags whether the system is diagonalized yet. 
    """

    def __init__(self,N,H=None):
        self.C = get_symplectic_matrix(N)
        if not type(H)==ndarray:
            self.H = zeros((2*N,2*N))
        else:
            self.H =H
            
        self.N = N 
        self.diagonalized = 0
        self.E    = None
        self.V    = None
        self.Vinv = None
    def get_x(self,n):
        """
        Get vector that corresponds to x-operator of mode n

        Parameters
        ----------
        n : int, mode

        Returns
        -------
        out : 2N vector of floats, vector representing operator. out[2*n]=1, while all other entries are zero

        """
        out = zeros(2*self.N)
        out[2*n]=1
        return out
    
    def get_p(self,n):
        """
        Get vector that corresponds to p-operator of mode n

        Parameters
        ----------
        n : int, mode

        Returns
        -------
        out : 2N vector of floats, vector representing operator. out[2*n+1]=1, while all other entries are zero

        """
        
        out = zeros(2*self.N)
        out[2*n+1]=1
        return out
    
    def diagonalize(self):
        """
        Diagonalize CH

        Returns
        -------
        [E,V,Vinv]
            Eigensystem (see above). Stores them as the object properties

        """
        [E,Vl,V] = eig(self.C@self.H,left=1,right=1)
        
        # New block
        AS = argsort(real(E))
        E = E[AS]
        V = V[:,AS]
        Vl = Vl[:,AS]
        
        
        self.E = E
        self.V = V
        Norm  = sum(V*Vl.conj(),axis=0,keepdims=True)
        Vl1 = Vl/Norm.conj()
        Vinv = Vl1.conj().T
        
        assert amin(imag(E))>-1e-12,"Negative frequency enconutered. Lamb shift overpowrerd. "
        self.Vinv=Vinv
        self.diagonalized = True
        
        return [E,V,Vinv]
    
    def evolve_vector(self,v0,t,eigenbasis=False):
        """
        Evolve vector v0 with propagator G(t) (see pdf notes)

        Parameters
        ----------
        v0 : ndarray (2*N), complex. Initial vector
            DESCRIPTION.
        t : float. Argument of G, See above 
        eigenbasis : bool, optional
            indicate whether we v(t) and v(0) are represented in (right) eigenbasis of G or not. The default is False.

        Returns
        -------
        TYPE
            v(t) = G(t)v0

        """
    
        assert self.diagonalized, "system must be diagonalized"
        
        if not eigenbasis:
            
            v0_eb = self.Vinv @ v0
            
            v_eb = exp(-2j*self.E*t)*v0_eb 
            
            return V@v_eb
    
        else:
            return exp(-2j*self.E*t)*v0
                 
    def evolve_matrix(self,mat,t1,t2,eigenbasis=False):
        """
        Evolve matrix mat with G: i.e., compute M(t1,t2) = G(t1) mat G(t2)^T
        

        Parameters
        ----------
        mat : ndarray, 2Nx2N, complex
            initial conditions
        t1,t2 : floats. Time arguments

        eigenbasis : bool, optional
            DESCRIPTION. Indicate whether M(t1,t2) and mat are represented in (right) eigenbasis of G.

        Returns
        -------
        
            M(t1,t2), ndarray 2Nx2N, complex.

        """
        
        assert self.diagonalized, "system must be diagonalized"
        
        if not eigenbasis:
            mat_eb = self.Vinv@mat@self.Vinv.T #NotImplementedError
            V1 = exp(-2j*self.E*t1).reshape((2*self.N,1))
            V2 = exp(-2j*self.E*t2).reshape((1,2*self.N))
            mat1_eb = V1*mat*V2
            return V@mat1_eb@(V.T)
        
        
        else:
            V1 = exp(-2j*self.E*t1).reshape((2*self.N,1))
            V2 = exp(-2j*self.E*t2).reshape((1,2*self.N))

            return V1*mat*V2
            
    def get_vacuum_correlation(self):
        """ 
        Get the correlation matrix in the vacuum state , 0.5*(1+C)
        """
        
        return 0.5*(eye(self.N*2)+self.C)
    
    def get_2point_function(self,K0,in_eigenbasis=False,operator_array=None):
        """
        get 2-point correlation function f(t1,t2) = <O_m(t1)O_n(t2)> as a method, for operators {O_n} stored in operator_array, and given initial condition <A_m(0)A_n(0)>=K0.

        Parameters
        ----------
        K0 : ndarray, 2Nx2N, complex. Initial condition
        in_eigenbasis : bool, optional
            indicate whether K and K0 are represented in eigenbasis. The default is False.
        operator_array : ndarray (2N,k), optional
            the k operators to be probed. Each operator O_n is a linear combination of the bosonic operators of the system: O_m = \sum_k operator_array[k,m]A_k.
            Is not specifiied, k=2*N, and O_n = A_n.

        Returns
        -------
        f(t1,t2) : method. Function that returns the matrix f(t1,t2)= <O_m(t1)O_n(t2)> 

        """
        if not self.diagonalized:
            self.diagonalize()
            
        M_eb = self.Vinv@K0@(self.Vinv.T)
   

        
        if in_eigenbasis:
            if type(operator_array)==ndarray:
                raise NotImplementedError("Does not work with operator_array")
            def f(t1,t2):
    
                return self.evolve_matrix(M_eb, t1, t2,eigenbasis=True)
            
        else:        
            if not type(operator_array)==ndarray:
                operator_array = eye(2*self.N)
            
            Q = operator_array.T@self.V            
            
            def f(t1,t2):
                """
                Correlation function
                Returns the matrix f(t1,t2)= <O_m(t1)O_n(t2)> 
                
                """
                return Q@self.evolve_matrix(M_eb, t1, t2,eigenbasis=True)@(Q.T)
            
            
            return f
            
        return f
    
    def get_1point_function(self,x0,in_eigenbasis=False,operator_array=None):
        """
        get 1-point correlation function f(t) = <O_m(t))> as a method, for operators {O_n} stored in operator_array, and given initial condition <A_m(0)>=x0.


        Parameters
        ----------
        x0 : ndarray (2N), float
            Initial condition (See above)
        in_eigenbasis : 
            not implemented. The default is False.
        operator_array :  ndarray (2N,k), optional
            the k operators to be probed. Each operator O_n is a linear combination of the bosonic operators of the system: O_m = \sum_k operator_array[k,m]A_k.
            Is not specifiied, k=2*N, and O_n = A_n.

        Returns
        -------
        f: method.
            function f(t) that returns ndarray (k), such that f(t)[n] <O_n(t)>

        """
        if not self.diagonalized:
            self.diagonalize()
        

        
        x0_eb = self.Vinv@x0
        if in_eigenbasis:
            if type(operator_array)==ndarray:
                raise NotImplementedError("Nontrivial operator array not implemented for eigenbase solution")
                    
            def f(t):
                return self.evolve_vector(x0_eb,t,eigenbasis=True)        
        
        else:
            if not type(operator_array)==ndarray:
                operator_array = eye(2*self.N)
        
            Q = operator_array.T@self.V 
            def f(t):
                return Q@self.evolve_vector(x0_eb,t,eigenbasis=True)       
        return f
    
    def get_spectral_function(self,K0,operator_array=None,interpolate=False):
        """
        Compute spectral function of operators {O_n} for time-indpendent system.
        The spectral function, J(\omega) is the fourier transform of the correlation function K(t+t_0,t_0) = <O_m(t+t_0)O_n(t_0)> for large t_0
        See pdf notes for definition and properties. 
        
        The spectral function takes the form J(\omega) = \sum_n M_n \delta(\omega-E_n). M_n and E_n are returned by this method (see below)
        
        Parameters
        ----------
        K0 : ndarray(2N,2N), complex. initial condition of system, <A_m(0)A_n(0)> = K0[m,n]
        
        
        operator_array : ndarray(2N,k), optional
            Operators O_n to be probed. Each operator O_n is a linear combination of the bosonic operators of the system: O_m = \sum_k operator_array[k,m]A_k.
            If not specifiied, k=2*N, and O_n = A_n.

        Returns
        -------
        
        [E,Weights]
            
            E: ndarray (2N) of floats. Frequencies E_n.
            Weights: ndarray (2N,k,k). Weights[n,:,:] = M_n
            
        """
        if not self.diagonalized:
            self.diagonalize()
    
        if not type(operator_array)==ndarray:
            operator_array = eye(2*self.N)
        
        Q = operator_array.T@self.V 
        
        
        M_sf_eb = self.Vinv@K0@(self.Vinv.conj().T)
        
    

        X= diag(diag(M_sf_eb))
        M_out = array([Q[:,n:n+1]@Q[:,n:n+1].conj().T*M_sf_eb[n,n] for n in range(0,2*self.N)])
        
        if not interpolate:
            
            return [2*self.E,M_out]
        
        else:
            dE = 0*self.E
            dE[1:-1] = 0.5*(self.E[2:]-self.E[:-2])
            dE[0] = self.E[1]-self.E[0]
            dE[-1] = self.E[-1]-self.E[-2]

            dE = dE*2
            K = interp1d(2*self.E,M_out/dE,kind="linear",axis=0,fill_value=0,bounds_error=0)
            return K 
        
    def get_ho_array(self,NP):
        """
        x=sqrt(0.5)*(b+b^*)
        p=-1j sqrt(0.5) (b-b^*)

        Parameters
        ----------
        NP : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        assert prod(NP)<1e6,"Hilbert space too large"
       
        # global S
        S = ho.ho_array(NP)
        
        Ind_r,Ind_c = where(abs(self.H)>1e-10)
        
        N_nnz = len(Ind_r)
        H_out = S.H
        for x in range(0,N_nnz):
            
            r = Ind_r[x]
            c = Ind_c[x]
            
            w = self.H[r,c]
            
            osc_r = r//2
            osc_c = c//2
            
            p_r   = r%2
            p_c   = c%2
            
            if p_r:
                op_r = S.get_p(osc_r)
            else:
                op_r = S.get_x(osc_r)
                
            if p_c:
                op_c = S.get_p(osc_c)
            else:
                op_c = S.get_x(osc_c)
                
  
            S.H += w*op_r@op_c
            
        return S

    def truncate(self,nlist):
          """
          Get trunctated system by keeping only oscillators in nlist
          """
          
          Out = gaussian_system(len(nlist))
    
          indices = zeros(2*len(nlist),dtype=int)
          for n in range(0,len(nlist)):
              indices[2*n]   = 2*nlist[n]
              indices[2*n+1] = 2*nlist[n]+1
              
          Out.H = self.H[:,indices][indices,:]
          
          return Out
    
class discrete_bath(gaussian_system):
    """
    Finite bath of oscillatgors with frequencies specified by wlist.  
    
    The bath comes with a bath operator (self.B) which has the spectral function given by spetral_function. 
    
    the bath operator is given by the x-coordinate of each oscillator, 
    
    B = \sum_{n=1}^N \sqrt{\Delta \omega_n S(\omega_n)} X_n
    
    where X_n = A_{2n} (see gaussian_system doc)
    
    Here S(omega) is the spectral function given as input argument    
        

    Parameters
    ----------
    wlist : frequencies of bath oscillators

    spectral_function : spectral function of bath observable.

    Returns
    -------
    None.

    """
    def __init__(self,wlist,spectral_function):

        gaussian_system.__init__(self,len(wlist))
        
        self.wlist = wlist 
        self.spectral_function = spectral_function
        self.B  = self.get_B()
        self.H = self.get_bath_hamiltonian()
                        
        
        



    def get_bath_hamiltonian(self):
        """
        Get bath Hamiltonian

        Returns
        -------
        H : ndarray of floats (2N,2N). 
        Hamiltonian of bath. Diagonal 

        """
        H = zeros((2*self.N,2*self.N))
        

        for n in range(0,self.N):
            H[2*n,2*n]      = 0.5*self.wlist[n]
            H[2*n+1,2*n+1]  = 0.5*self.wlist[n]

        return H
    
    def get_B(self):
        """ 
        Get "B" operator, which is coupled to system, using spectral function. 
        Weighted with frequency spacing. See code for how the spacing is determined
        """
        
        B = zeros((2*self.N))

        for n in range(0,self.N):
            if n>0 and n<self.N-1:
                
                dw = 0.5*(self.wlist[n+1]-self.wlist[n-1])
            elif n==0:
                dw = (self.wlist[1]-self.wlist[0])
            else:
                dw = (self.wlist[self.N-1]-self.wlist[self.N-2])

                
                
                
            B[2*n]  =  sqrt(dw*self.spectral_function(self.wlist[n]))
            
        return B
    
    def get_equilibrium_correlation_matrix(self,temperature):
        """
        Get correlation matrix in the equilibrium state of the system at a given temperature

        Parameters
        ----------
        temperature : float

        Returns
        -------
        Mat : Equilibrium correlation matrix <A_m A_n>_{eq} at the given temperature.
        """
        out = self.get_vacuum_correlation()
        m1 = bose_einstein(self.wlist,temperature)
        for n in range(0,self.N):
           out[2*n,2*n]+=m1[n]
           out[2*n+1,2*n+1]+=m1[n]
           
        return out
    
    
        
    def connect_with_system(self,System,X_sys):
        """ Connect system and bath with system-bath coupling X_sys * Bath.B.
        Returns combined_system object
        """
        
        return combined_system(System,self,X_sys)
                               
    def old_connect_with_system(self,System,X_sys):
        raise DeprecationWarning("This method is deprecated")
        # d_sys = shape(H_sys)[0]
        d_sys = 2*System.N
        # assert d_sys%2==0,"dimension of h_sys must be even"
        n_sys = System.N
        H_sys = System.H
        n_tot = self.N+n_sys
        d_tot = d_sys+2*self.N
        H_out = zeros((2*n_tot,2*n_tot))
        
        H_out[:d_sys,:d_sys] = H_sys
        H_out[d_sys:,d_sys:] = self.H
        
        H_out[:d_sys,d_sys:] = 0.5*outer(X_sys,self.B)
        H_out[d_sys:,:d_sys] = 0.5*outer(self.B,X_sys)
        
        out = gaussian_system(n_tot)
        out.H = H_out
        
        return out
    
    def get_bath_spectral_function(self,temperature):
        """
        Get spectral function of bath in operator arrray at a given temperature. (see get_spectral_function method for more detials)

        """
        K0 = self.get_equilibrium_correlation_matrix(temperature)
        [E,W] =  self.get_spectral_function(K0,operator_array=self.B.reshape((self.N*2,1)))
        W = W[:,0,0]
        
        return [E,W]
    
class combined_system(gaussian_system):
    """
    Object which consists of system and bath. H = H_Sys + H_bath + Xsys * Bath.B
    Has methods for time-evolution and spectral functions which distinguishes between system and bath.
    
    First 2*Nsys entries in Hamiltonian correspond to system degrees of freedom. The remaining correspond to bath degrees of freedom
    
    I.e., 
    
    H = [   System.H         0.5*Xsys@Bath.B.T
            0.5.Bath.B@Xsys.T  Bath.H]
    
    
    Parameters
    ----------
    system : gaussian_system object
        The system which is connected with the bath

    bath  : discrete_bath object
        The bath
    
    X_sys : ndarray(2*Nsys), complex
        The system operator which is coupled to the bath.B
        
    Special properties
    ------------------
    
    Nsys  : int
        number of oscillators in system
    Nbath : int
        number os oscillators in bath.
    
    system : gaussian_system
        object of system
    bath   : discrete_bath
        object of bath
    
    """
    
    def __init__(self,system,bath,X_sys):
        Nsys = system.N
        Nbath = bath.N
        gaussian_system.__init__(self,Nsys+Nbath)
        
        self.system = system
        self.bath   = bath
        self.Nsys = Nsys
        self.Nbath = Nbath
        d_sys = 2*Nsys
        d_bath = 2*Nbath
        
        H_out = zeros((2*self.N,2*self.N))
        
        H_out[:d_sys,:d_sys] = self.system.H
        H_out[d_sys:,d_sys:] = self.bath.H
        
        H_out[:d_sys,d_sys:] = 0.5*outer(X_sys,self.bath.B)
        H_out[d_sys:,:d_sys] = 0.5*outer(self.bath.B,X_sys)
        
        self.H = H_out
    
    def get_systems_2point_function(self,K0_sys,bath_temperature=0,operator_array=None):
        """
        get 2-point correlation function f(t1,t2) = <O_m(t1)O_n(t2)> as a method, for system operator {O_n} stored in operator_array, given initial condition for system <O_m(0)O_n(0)> = K0]m,n] and initial bath temperature bath_temperature

        Parameters
        ----------
        K0 : ndarray, 2Nsysx2Nsys, complex. Initial condition

        operator_array : ndarray (2Nsys,k), optional
            the k system operators to be probed. Each operator O_n is a linear combination of the bosonic operators of the system: O_m = \sum_k operator_array[k,m]A_k.
            Is not specifiied, k=2*N, and O_n = A_n.

        Returns
        -------
        f(t1,t2) : method. Function that returns the matrix f(t1,t2)[m,n]= <O_m(t1)O_n(t2)> 
        """
        if not type(operator_array)==ndarray:
            operator_array = eye(2*self.Nsys)
        
        K0_bath = self.bath.get_equilibrium_correlation_matrix(bath_temperature)
        
        K0 = zeros((2*self.N,2*self.N),dtype=complex)
        
        K0[:2*self.Nsys,:2*self.Nsys] = K0_sys
        K0[2*self.Nsys:,2*self.Nsys:] = K0_bath
        
        self.diagonalize()


        f = self.get_2point_function(K0,in_eigenbasis=True)
        
        Q = operator_array.T@self.V[:2*self.Nsys,:]
        
        def g(t1,t2):
            return Q@f(t1,t2)@(Q.T)
        
        return g
    
        
    def get_systems_1point_function(self,x0_sys,operator_array=None):
        """
        get 1-point correlation function f(t) = <O_m(t))> as a method, for system operators {O_n} stored in operator_array, and given initial condition <A_m(0)>=x0.


        Parameters
        ----------
        x0 : ndarray (2Nsys), float
            Initial condition (See above)
        in_eigenbasis : 
            not implemented. The default is False.
        operator_array :  ndarray (2Nsys,k), optional
            the k operators to be probed. Each operator O_n is a linear combination of the bosonic operators of the system: O_m = \sum_k operator_array[k,m]A_k.
            Is not specifiied, k=2*N, and O_n = A_n.

        Returns
        -------
        f: method.
            function f(t) that returns ndarray (k), such that f(t)[n] <O_n(t)>

        """
        if not type(operator_array)==ndarray:
            operator_array = eye(2*self.Nsys)
            
        x0 = zeros((2*self.N))
        x0[:self.Nsys*2]=x0_sys

        Z=operator_array
        f = self.get_1point_function(x0,in_eigenbasis=True)
        
        
            
        
        Q = operator_array.T@self.V[:2*self.Nsys,:]
        
        def g(t):
            return Q@f(t)
        
        
        return g
        
    def get_systems_spectral_function(self,temperature,operator_array=None,interpolate=False):
        """
        Compute spectral function of system operators {O_n}, given some temperature of the bath.
        The spectral function, J(\omega) is the fourier transform of the correlation function K(t+t_0,t_0) = <O_m(t+t_0)O_n(t_0)> for large t_0
        See pdf notes for definition and properties. 
        
        The spectral function takes the form J(\omega) = \sum_n M_n \delta(\omega-E_n). M_n and E_n are returned by this method (see below)
      
        Parameters
        ----------
        temperature : float,
            Temperature of bath
        
        operator_array : ndarray(2Nsys,k) of floats, optional
            Operators O_n to be probed. Each operator O_n is a linear combination of the bosonic system operators of the system: O_m = \sum_k operator_array[k,m]A_k.
            If not specifiied, k=2*N, and O_n = A_n.

        """
        if not type(operator_array)==ndarray:
            operator_array = eye(2*self.Nsys)
        
        nop = shape(operator_array)[1]
        K0_bath = self.bath.get_equilibrium_correlation_matrix(temperature)
        K0_sys  = self.system.get_vacuum_correlation()
        
        K0 = zeros((2*self.N,2*self.N),dtype=complex)
        
        K0[:2*self.Nsys,:2*self.Nsys] = K0_sys
        K0[2*self.Nsys:,2*self.Nsys:] = K0_bath
        
        
        operator_array=concatenate((operator_array,zeros((2*self.Nbath,nop))))
        
        return self.get_spectral_function(K0,operator_array,interpolate=interpolate)  
              
class floquet_system(gaussian_system):
    """
    Floquet system. 
    Same as gaussian system, but now the Hamiltonian depends on time with some periodicity.
    Here self.H is the effective Hamiltonian. 
    Solve Floquet problem with drive resolution specified by Nres
    """
    def __init__(self,H_method,driving_period,Nres = 300):
        assert callable(H_method);"H must be callable"
        assert amax(abs(H_method(driving_period)-H_method(0))<1e-10), "H must have periodicity identical to driving_period"
        
        N = H_method(0).shape[0]//2
        self.H_method = H_method
        self.T = driving_period
        gaussian_system.__init__(self,N)
        # self.diagonalize = 
        self.Nres = Nres
        self.dt = self.T/self.Nres

        self.floquet_operator_found = False
        
        self.floquet_operator = None
        self.V = None
        self.E = None
        self.Vinv = None
        self.H = None
                
    def floquet_restrict(self,method):
        """Restrict functions  of times to functions of integer multiples of driving periods
        """
        def f1(*args):
            """Give evolution after n driving periods
            """
            
            assert prod([type(x)==int for x in args]),"Arguments must be integers"
            
            times = tuple([x*self.T for x in args])

            return method(*times)
        
        return f1   
        
    def solve_floquet_operator(self):
        
        U = eye(self.N*2)
        
        
        for n in range(0,self.Nres):
            t = n*self.dt

            Mat = real(-2j*self.C@self.H_method(t))
            
            U = expm(Mat*self.dt)@U
            
        self.floquet_operator_found = True
        self.floquet_operator = U
        
        
        return U
    
    def diagonalize(self):
        if not self.floquet_operator_found:
            self.solve_floquet_operator()
            
        
        
        [PH,Vl,V] = eig(self.floquet_operator,left=1,right=1)
        E = 1j*log(PH)/(2*self.T)
        AS = argsort(real(E))
        E = E[AS]
        V = V[:,AS]
        self.E = E
        self.V = V
        
        Norm  = sum(V*Vl.conj(),axis=0,keepdims=True)
        Vl1 = Vl/Norm.conj()
        Vinv = Vl1.conj().T
        
        assert amin(imag(E))>-1e-12,"Negative frequency enconutered. Lamb shift overpowrerd. "
        self.Vinv=Vinv
        self.diagonalized = True
        
        self.H = self.Vinv@diag(self.E)@self.V
        
        
        return [E,V,Vinv]
    
    def get_1point_function(self,x0,in_eigenbasis=False,operator_array = None):
        if not self.diagonalized:
            self._diagonalize()
        
        f0 = gaussian_system.get_1point_function(self,x0,in_eigenbasis=in_eigenbasis,operator_array = operator_array)
        
        return self.floquet_restrict(f0)
    
        # def f1(n):
        #     """Give evolution after n driving periods
        #     """
        #     assert type(n)==int,"Argument must be integer"

        #     return f0(n*self.T)

        # return f1

    def get_2point_function(self,K0,in_eigenbasis=False,operator_array=None):
        if not self.diagonalized:
            self.floquet_diagonalize()
        
        f0 = gaussian_system.get_2point_function(self,K0,in_eigenbasis=in_eigenbasis,operator_array=operator_array)
        
        return self.floquet_restrict(f0)
        # def f1(n1,n2):
        #     """Give evolution after n driving periods
        #     """
        #     assert type(n1)==int and type(n2)==int,"Arguments must be integers"

        #     return f0(n1*self.T,n2*self.T)

        # return f1

    

    def get_spectral_function(self, K0):
        raise NotImplementedError("Spectral function not defined for Floquet systems.")
    
    
  
              
class combined_floquet_system(floquet_system,combined_system):
    """Floquet system connected to (time-independent) bath
    """
    
    def __init__(self,fs,bath,X_sys):
        assert type(fs)==floquet_system
        assert type(bath) ==discrete_bath
        self.system = fs
        self.bath   = bath
        
        Nsys = self.system.N
        Nbath = self.bath.N
        self.Nsys = Nsys
        self.Nbath = Nbath
        self.N = Nsys+Nbath
        self.X_sys  = X_sys
        
        get_hamiltonian = self.get_full_system_hamiltonian_method()

        combined_system.__init__(self,fs,bath,X_sys)
        floquet_system.__init__(self,get_hamiltonian,self.system.T,Nres=fs.Nres)
        
 
    
    def get_full_system_hamiltonian_method(self):
        d_sys = 2*self.Nsys
        d_bath = 2*self.Nbath
        
        H_method = self.system.H_method
        
        def get_hamiltonian(t):
            
            
            H_out = zeros((2*self.N,2*self.N))
            
            H_out[:d_sys,:d_sys] = H_method(t)
            H_out[d_sys:,d_sys:] = self.bath.H
            
            H_out[:d_sys,d_sys:] = 0.5*outer(self.X_sys,self.bath.B)
            H_out[d_sys:,:d_sys] = 0.5*outer(self.bath.B,self.X_sys)
            
            return H_out

        return get_hamiltonian 
    
    def get_systems_1point_function(self, x0_sys, operator_array=None):
        f0 = combined_system.get_systems_1point_function(x0_sys,operator_array=operator_array)
        return self.floquet_restrict(f0)
        
    def get_systems_2point_function(self,K0_sys,bath_temperature=0,operator_array=None):
        f0 = combined_system.get_systems_2point_function(K0_sys,bath_temperature=bath_temperature,operator_array=operator_array)
        return self.floquet_restrict(f0)       
        
    def get_systems_spectral_function(self, temperature):
        raise NotImplementedError("Spectral function not defined for Floquet systems.")


class purcell_filter(gaussian_system):
    """

    Purcell filter with Hamiltonian 

    \hat{H} = \sum_{n=1}^{N-1} J \hat{x}_n\hat{x}_{n+1} + \sum_n \omega \hat b^\dagger_n \hat b_n 

    Here \hat x_n = \sqrt{0.5}(\hat b_n + \hat b^\dagger_n), 
    """

    def __init__(self,N,omega,J):
        gaussian_system.__init__(self,N)
        
        for n in range(0,N):

            self.H[2*n,2*n]     = omega /2
            self.H[2*n+1,2*n+1] = omega/2
            
            if n<N-1:
                    
                self.H[2*n,2*n+2] = J/2
                self.H[2*n+2,2*n] = J/2


        self.omega = omega
        self.J     = J 



        
if __name__=="__main__":
    """
    Demonstrate module on array of oscillators coupled to an ohmic bath. 
    Compute dynamics and spectral function of the system. 
    """
    ### Oscillator to track    
    nosc=0

    ### Define bath 
    sigma = 2
    N_bath=100 # Number of oscillators in the bath
    wmax = 5*sigma
    dw = wmax/N_bath
    # w0 = 2
    wlist = dw*(1+arange(N_bath))
    bath_temperature= 1

    gamma=0.1
    def S(w):
        return w*exp(-w**2/(2*sigma**2))

    Bath = discrete_bath(wlist,S)
    

    ### Define system
    N_s = 2
    H_s = zeros((2*N_s,2*N_s))
    
    w = array([1,2.2])+2.5
    J = 0.3
    for n in range(0,N_s):
        H_s[2*n,2*n] = w[n]/2
        H_s[2*n+1,2*n+1] = w[n]/2
        
        try:
            H_s[2*n,2*(n+1)]=J/2
            H_s[2*(n+1),2*n]=J/2
            
        except IndexError :
            pass
   
    System = gaussian_system(N_s,H=H_s)
    
    ### Floquet_system
    V = H_s*0 
    V[1,0] = 1
    V[0,1] = 1
    
    V= V*0
    def H_method(t):
        return H_s + V*cos(2*pi*t)
    
    FS = floquet_system(H_method,1)
    S  = gaussian_system(N_s,H=H_s)
    
    Q = S.get_ho_array((10,10))
    S.diagonalize()
    [E,V] = eigh(Q.H.toarray())
    raise ValueError
    FS.diagonalize()
    
    F1 = FS.get_1point_function(FS.get_x(0))
    F2 = FS.get_2point_function(FS.get_vacuum_correlation())
    
    X0 = FS.get_x(0)
    CFS = combined_floquet_system(FS, Bath, X0)
    CS  = combined_system(S,Bath,X0)
    CS.diagonalize()
    CFS.diagonalize()

        
    # Combine system and baths
    Full  = Bath.connect_with_system(System,System.get_x(0)*sqrt(gamma))
   
    
    # Define what we want to measure     
    oa = array([System.get_x(n) for n in range(0,4)]).T
    
    
    # initialize expectation values    
    x0 = Full.system.get_x(N_s-1)*3

    OPF = Full.get_systems_1point_function(x0,operator_array=oa)
    
    K0 = Full.system.C*0.5
    K0[0,0]+=9
    
    CF = Full.get_systems_2point_function(K0,bath_temperature=bath_temperature,operator_array=oa)

    

    dt = 0.5
    tlist =arange(0,200,0.3)
    NT = len(tlist)
    outlist_1 = zeros((NT,4,4),dtype=complex)
    outlist_2 = zeros((NT,4,4),dtype=complex)
    outlist_3 = zeros((NT,4),dtype=complex)

    nt=0
    t0=40
    for t in tlist:
        outlist_1[nt] = CF(t,t0)
        outlist_2[nt] = CF(t,t)
        outlist_3[nt] = OPF(t)
        
        nt+=1
        
        
    figure(1)
    
    plot(tlist,real(outlist_1)[:,nosc,nosc],'b')
    plot(tlist,imag(outlist_1)[:,nosc,nosc],'r')
    legend(["Real part","Imaginary part"])
    xlabel("t")
    title(f"Correlation function $\\langle x_{nosc+1}(t)x_{nosc+1}(t_0)\\rangle$; $t_0={t0}$")
    plot([t0,t0],[-5,1.2*amax(abs(outlist_1))],":k")
    text(t0,1.25*amax(abs(outlist_1)),"$t_0$",ha="center",va="bottom")
    ylim(-1.2*amax(abs(outlist_1)),1.5*amax(abs(outlist_1)))
    
    figure(2)
        
    plot(tlist,sqrt(real(outlist_2))[:,nosc,nosc],'b')
    plot(tlist,real(outlist_3)[:,nosc],'r')
    xlabel("t")
    title(f"Evolution of oscillator {nosc+1} where $\\langle x_{N_s}(0)\\rangle = 3$; $T=$"+f"{bath_temperature}"+" $\gamma=$"+f"{gamma}")
    legend(["$\sqrt{\\langle x^2(t)\\rangle}$",f"$\\langle x(t)\\rangle$"])
    
    [Freqs,Weights] = Full.get_systems_spectral_function(bath_temperature)
        
    figure(3)
    AS = argsort(Freqs)
    Freqs = Freqs[AS]
    Weights = Weights[AS]
    NF =len(Freqs)
    
    plot(Freqs,real(Weights[:,nosc,nosc]),'.-',markersize=5)
    xlabel("Frequency")
    title(f"Spectral function of $x_{1+nosc}$;"+", $T_{\\rm bath}=$"+f"{bath_temperature}; "+"$\gamma=$"+f"{gamma}")
    
    vmax = amax(real(Weights[:,nosc,nosc]))
    plot([w[nosc],w[nosc]],[0,vmax*1.1],":k",lw=0.7)
    text(w[nosc],1.115*vmax,"$\omega"+f"_{nosc+1}$",ha="center",va="bottom")
    
    plot([-w[nosc],-w[nosc]],[0,vmax*1.1],":k",lw=0.7)
    text(-w[nosc],1.115*vmax,"$-\omega"+f"_{nosc+1}$",ha="center",va="bottom")
    ylim(0,1.3*vmax)





    