#!/usr/bin/env python3
"""
GPLv3 2020 Miguel Aguilera

This code runs a simualtion of the kinetic Ising models form an initial
state for T steps repeated over R trials. The results are used for reference
in the forward Ising model and as input for the inverse Ising model.
All simulations used pre-generated parameters
An example of the data generated from this file can be found at XXX.
"""

import context
from kinetic_ising import ising
import numpy as np
from sys import argv


size = 512                 # Network size
R = 1000000                # Repetitions of the simulation
H0 = 0.5                   # Uniform distribution of fields parameter
J0 = 1.0                   # Average value of couplings
Js = 0.1                   # Standard deviation of couplings

I = ising(size)

B = 21                    # Number of values of beta
T = 2**7                  # Number of simulation time steps

# Set critical inverse temperature value
if H0 == 0.5 and Js == 0.1:
    beta0 = 1.1108397534245904
else:
    print('Undefined beta0')
    beta0 = 1

# Define reference values of beta, normalized by the critical temperature
betas = 1 + np.linspace(-1, 1, B) * 0.3

# Load network parameters
filename = 'data/parameters_H0-' + \
    str(H0) + '-J0-' + str(J0) + '-Js-' + str(Js) + '-N-' + str(size) + '.npz'
data = np.load(filename)
H_init = data['H']
J_init = data['J']

# Run for each value of beta
for ib in range(len(betas)):
    beta_ref = round(betas[ib], 3)
    beta = beta_ref * beta0
    print(beta_ref, str(ib) + '/' + str(len(betas)), H0, J0, Js, size)

    I.H = beta * H_init.copy()
    I.J = beta * J_init.copy()

    m_exp = np.zeros((size, T))
    m_exp_prev = np.zeros((size, T))
    C_exp = np.zeros((size, size, T))
    D_exp = np.zeros((size, size, T))

    # Initial state is all ones
    s0 = np.ones(size)

    print('generate data')
    # Repeat for R repetitions
    for rep in range(R):
        I.s = s0.copy()
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
            C_exp[range(size), range(size), t] += (1 - np.tanh(h)**2) / R
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
    filename = 'data/data-H0-' + str(H0) + '-J0-' + str(J0) + '-Js-' + str(
        Js) + '-N-' + str(size) + '-R-' + str(R) + '-beta-' + str(beta_ref) + '.npz'
    np.savez_compressed(
        filename,
        m=m_exp,
        C=C_exp,
        D=D_exp,
        H=I.H,
        J=I.J,
        s0=s0,
        beta_c=beta0)
