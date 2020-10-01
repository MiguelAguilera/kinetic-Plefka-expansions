#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:45:08 2018

@author: maguilera
"""

from mf_ising import mf_ising
import numpy as np
from matplotlib import pyplot as plt

plt.rc('text', usetex=True)
font = {'size': 15}
plt.rc('font', **font)
plt.rc('legend', **{'fontsize': 12})

size = 512
save = False

gamma1 = 0.5
gamma2 = 0.1


I = mf_ising(size)

B = 21
T = 2**8


iu1 = np.triu_indices(size, 1)

m_mean = np.zeros(B)
m_mean1 = np.zeros(B)
D_mean = np.zeros(B)
D_mean1 = np.zeros(B)
C_mean = np.zeros(B)
C_mean1 = np.zeros(B)


# Set critical inverse temperature value
if gamma1 == 0.5 and gamma2 == 0.1:
    beta0 = 1.099715
else:
    print('Undefined critical beta')
    beta0 = 1

# Define reference values of beta, normalized by the critical temperature
betas = 1 + np.linspace(-1, 1, B) * 0.3

# Generate random parameter seed
H_seed = np.random.rand(size)
J_seed = np.random.randn(size, size)
# Generate random parameters
H0 = gamma1 * (H_seed * 2 - 1)
J0 = 1 / size + gamma2 * J_seed / np.sqrt(size)

# Run the network for different inverse temperatures
for ib in range(len(betas)):
    beta_ref = betas[ib]
    beta = beta_ref * beta0
    print(beta_ref, str(ib) + '/' + str(len(betas)), gamma1, gamma2, size)
    I.H = beta * H0
    if np.mean(I.H) < 0:        # Ensure that the network is kept on the positive symmetry-breaking side
        I.H *= -1
    I.J = beta * J0

    # Initialize statistical moments
    m_exp = np.zeros((size, T))
    m_exp_prev = np.zeros((size, T))
    C_exp = np.zeros((size, size, T))
    D_exp = np.zeros((size, size, T))

    # Initial state on the positive symmetry-breaking side
    s0 = np.ones(size)
    I.initialize_state(s0)

    # Run simulation with Plefka[t] to check convergence
    for t in range(T):
        I.update_P2_t_o2()
        print(beta_ref,t,np.mean(I.m), np.mean(I.C[iu1]),np.mean(I.D))
        if t == T // 2:
            C_mean1[ib] = np.mean(I.C[iu1])
            D_mean1[ib] = np.mean(I.D)
            m_mean1[ib] = np.mean(I.m)
    # Indices for upper triangle of the (symmetric) correlation matrix C
    iu1 = np.triu_indices(size, 1)
    m_mean[ib] = np.mean(I.m)
    C_mean[ib] = np.mean(I.C[iu1])
    D_mean[ib] = np.mean(I.D)
    print(m_mean[ib], C_mean[ib], D_mean[ib])


# Save parameters
filename='data/parameters_size-'+str(size)+'-gamma1-' + str(gamma1) +'-gamma2-' + str(gamma2) +'.npz'
if save:
    np.savez_compressed(filename, H0=H0, J0=J0)

# Plot results
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
plt.plot(betas, np.abs(m_mean), 'k')
plt.plot(betas, np.abs(m_mean1), 'k--')
plt.ylabel(r'$\langle m_{i,t} \rangle$', fontsize=18, rotation=0, labelpad=20)
plt.xlabel(r'$\beta/\beta_c$', fontsize=18)
plt.axis([np.min(betas), np.max(betas), 0, 1.05 * np.max(np.abs(m_mean))])
#plt.savefig('img/model-m_size-' + str(size) + '1.pdf', bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
plt.plot(betas, C_mean, 'k')
plt.plot(betas, C_mean1, 'k--')
plt.ylabel(r'$\langle C_{ik,t} \rangle$', fontsize=18, rotation=0, labelpad=20)
plt.xlabel(r'$\beta/\beta_c$', fontsize=18)
plt.axis([np.min(betas), np.max(betas), 0, 1.05 * np.max(np.abs(C_mean))])
#plt.savefig('img/model-C_size-' + str(size) + '1.pdf', bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
plt.plot(betas, D_mean, 'k')
plt.plot(betas, D_mean1, 'k--')
plt.ylabel(r'$\langle D_{il,t} \rangle$', fontsize=18, rotation=0, labelpad=20)
plt.xlabel(r'$\beta/\beta_c$', fontsize=18)
plt.axis([np.min(betas), np.max(betas), 0, 1.05 * np.max(np.abs(D_mean))])
#plt.savefig('img/model-D_size-' + str(size) + '-1.pdf', bbox_inches='tight')
plt.show()
