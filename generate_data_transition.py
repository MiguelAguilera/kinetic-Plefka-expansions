#!/usr/bin/env python3
"""
GPLv3 2020 Miguel Aguilera

This code is similar to "generate_data.py" but only saves the average
means and correlations (same-time and time-delayed), as well as the 
entropy production. It is used for the purpose of generating data 
with shorter simulations for more points of beta.
An example of the data generated from this file can be found at XXX.
"""

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
elif gamma1 == 0.5 and gamma2 == 0.1:
    beta0 = 1.1108397534245904
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

    m_exp = np.zeros((size))
    m_exp_prev = np.zeros((size))
    D_exp = np.zeros((size, size))
    

    # Initial state is all ones 
    s0 = np.ones(size)
    
    print('generate data')
    # Repeat for R repetitions
    for rep in range(R):
        # Run the simulation for T steps
        for t in range(T):
            s_prev = I.s.copy()              # Save previous state
            h = I.H + np.dot(I.J, s_prev)    # Compute effective field
            # Compute statistical moments
            if t==T-2:
                m_exp_prev += np.tanh(h) / R
            if t==T-1:
                m_exp += np.tanh(h) / R
                C_exp[:, :] += np.einsum('i,j->ij',
                                        np.tanh(h),
                                        np.tanh(h),
                                        optimize=True) / R
                
                C_exp[range(size),range(size)] += (1 - np.tanh(h)**2) / R
                D_exp+= np.einsum('i,j->ij',
                                        np.tanh(h),
                                        s_prev,
                                        optimize=True) / R
            I.ParallelUpdate()                # Update the state of the system
    for t in range(T - 1):
    # Substract product of means to compute covariances
    C_exp -= np.einsum('i,j->ij', m_exp, m_exp, optimize=True)
    D_exp -= np.einsum('i,l->il', m_exp, m_exp_prev, optimize=True)

    iu1 = np.triu_indices(size, 1)
    m_mean=np.mean(m_exp)
    C_mean=np.mean(C_exp[iu1])
    D_mean=np.mean(D_exp)
    sigma = np.sum(J*(D_exp-D_exp.T))
    
    # Save the evolution of statistical moments
    filename = 'data/results/data-transition-gamma1-' + str(gamma1) +'-gamma2-' + str(gamma2) + '-s-' + \
            str(size) + '-R-' + str(R) + '-beta-' + str(beta_ref) + '.npz'
    ndec=5     # Number of significative figures
    np.savez_compressed(filename, m_mean=m_mean, C_mean=C_mean, D_mean=D_mean, H=H, J=J, s0=s0, beta_c=beta0)
