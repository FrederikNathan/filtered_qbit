# ULE_and_gaussian_code
Python  modules for simulating arrays of Harmonic oscillators using the ULE, and using a gaussian solver. There are two indpendent families of modules (see below); one for solving the ULE, and one for the guassian solution. The gaussian code works only for quadratic systems, but is exact. The ULE works for any system (including systems with nonlinearities), but is approximate. 

The ULE solver works only through exact diagonalization of the Liouvillian so far. This solution method is feasilbe only for systems with Hilbert space dimension \lesssim 100. A stochastic solver would work for larger systems but is not implemented yet. 

The two solution methods are demonstrated in the script purcell_test.py (see below). 

The theory behind the code is written up in finite_gassian_system.pdf (2 pages) for the gaussian solution, and PhysRevB.102.115109.pdf (24 pages) for the ULE part. 

## Contents:

### basic.py
Module with some useful basic functins and variables

### finite_gaussian.py 
Module for exact solution of finite guassian systems and baths. 

### ho_array.py
Module for constructing Hamiltonians of arrays of Harmonic oscillators. Allows for nonlineariteis

### thermal_bath.py
Module for representing gaussian baths though spectral functions. Used to generate jump operators, Lamb shift, and to compute ULE timescales

### lindblad_solver.py
Module for solving lindblad form master equations in various ways

### vectorization.py
Module for vectorizing matrices (used in lindblad_solver to generate liouvillian matrix)

### purcell_test.py.
Script that simulates a system with a single oscillator coupled to an ohmic bath through a purcell filter. The script uses the gaussian method and the ULE, and compares the two methods. 

