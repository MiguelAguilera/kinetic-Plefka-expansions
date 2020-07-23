#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:45:08 2018

@author: maguilera
"""

from mf_ising import mf_ising
from kinetic_ising import ising
import numpy as np
from matplotlib import pyplot as plt
from sys import argv

if len(argv) < 3:
    print("Usage: " + argv[0] + " <network size>"  + " <repetitions>"+ " <gamma1>"+ " <gamma2>")
    exit(1)

size=int(argv[1])        # Network size
R=int(argv[2])            # Repetitions of the simulation
gamma1=float(argv[3])    # 
gamma2=float(argv[4])

I = ising(size)

B = 21                    # Number of values of beta
T = 2**7                  # Number of simulation time steps

# Set critical inverse temperature value
if gamma1 == 0.5 and gamma2 == 0.2:
    beta0 = 1.1451
elif gamma1 == 0.5 and gamma2 == 0.1:
    beta0 = 1.1123
elif gamma1 == 0.5 and gamma2 == 0.05:
    beta0 = 1.1041
elif gamma1 == 0.5 and gamma2 == 0.02:
    beta0 = 1.1019
elif gamma1 == 0.5 and gamma2 == 0.01:
    beta0 = 1.1015
else:
    print('Undefined beta0')
    beta0 = 1

# Define reference values of beta, normalized by the critical temperature
betas = 1 + np.linspace(-1, 1, B) * 0.3


def RoundToSigFigs_fp( x, sigfigs ):
    """
    Rounds the value(s) in x to the number of significant figures in sigfigs.
    Return value has the same type as x.

    Restrictions:
    sigfigs must be an integer type and store a positive value.
    x must be a real value.
    """
    xsgn = np.sign(x)
    absx = xsgn * x
    with np.errstate(divide='ignore'):
         R = 10**np.ceil(np.log10(absx))
    R[absx==0]=1

    return xsgn * R * np.around( absx/R, decimals=sigfigs)
    
    
# Load network parameters
filename='data/parameters_size-'+str(size)+'-gamma1-' + str(gamma1) +'-gamma2-' + str(gamma2) +'.npz'
data=np.load(filename)
H0=data['H0']
J0=data['J0']


# Run for each value of beta
for ib in range(len(betas)):
    beta_ref = round(betas[ib],3)
    beta = beta_ref * beta0
    print(beta_ref,str(ib)+'/'+str(len(betas)),gamma1,gamma2,size)
    
    I.H = beta * H0
    if np.mean(I.H)<0:
        I.H*=-1
    I.J= beta * J0

    J = I.J.copy()
    H = I.H.copy()

    print(J[0,0:10])
    print(J0[0,0:10]*beta)
    m_exp = np.zeros((size, T))
    m_exp_prev = np.zeros((size, T))
    C_exp = np.zeros((size, size, T))
    D_exp = np.zeros((size, size, T))
    

    # Initial state is all ones 
    s0 = np.ones(size)
    
    print('generate data')
    # Repeat for R repetitions
    for rep in range(R):
        m_exp_prev[:, 0] += I.s.copy() / R   # Mean value at the previous state
        # Run the simulation for T steps
        for t in range(T):
            s_prev = I.s.copy()              # Save previous state
            h = I.H + np.dot(I.J, s_prev)    # Compute effective field
            # Compute statistical moments
            m_exp[:, t] += np.tanh(h) / R
            C_exp[:, :, t] += np.einsum('i,j->ij',
                                        np.tanh(h),
                                        np.tanh(h),
                                        optimize=True) / R
            C_exp[range(size),range(size),t] += (1 - np.tanh(h)**2) / R
            D_exp[:, :, t] += np.einsum('i,j->ij',
                                        np.tanh(h),
                                        s_prev,
                                        optimize=True) / R
            I.ParallelUpdate()                # Update the state of the system
    for t in range(T - 1):
        m_exp_prev[:, t + 1] = m_exp[:, t]   # Mean value at the previous state
    # Substract product of means to compute covariances
    C_exp -= np.einsum('it,jt->ijt', m_exp, m_exp, optimize=True)     
    D_exp -= np.einsum('it,lt->ilt', m_exp, m_exp_prev, optimize=True)

    # Save the evolution of statistical moments
    filename = 'data/m-c-ts0-gamma1-' + str(gamma1) +'-gamma2-' + str(gamma2) + '-s-' + \
            str(size) + '-R-' + str(R) + '-beta-' + str(beta_ref) + '.npz'
    ndec=5     # Number of significative figures
    np.savez_compressed(filename, C=np.around(C_exp,ndec), m=np.around(m_exp,ndec), D=np.around(D_exp,ndec), H=H, J=J, s0=s0, beta0=beta0)
