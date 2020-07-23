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
from matplotlib import cm

plt.rc('text', usetex=True)
font = {'size': 15}
plt.rc('font', **font)
plt.rc('legend', **{'fontsize': 12})


size = 512

R = 500000
mode = 'c'
gamma1 = 0.5
gamma2 = 0.1

beta = 1.0


T = 128
iu1 = np.triu_indices(size, 1)

EmP0o2 = np.zeros(T + 1)
EmP1o2 = np.zeros(T + 1)
EmP1Co2 = np.zeros(T + 1)
EmP2o1 = np.zeros(T + 1)

ECP0o2 = np.zeros(T + 1)
ECP1o2 = np.zeros(T + 1)
ECP1Co2 = np.zeros(T + 1)
ECP2o1 = np.zeros(T + 1)

EDP0o2 = np.zeros(T + 1)
EDP1o2 = np.zeros(T + 1)
EDP1Co2 = np.zeros(T + 1)
EDP2o1 = np.zeros(T + 1)


mP0o2_mean = np.ones(T + 1)
mP1o2_mean = np.ones(T + 1)
mP1Co2_mean = np.ones(T + 1)
mPexp_mean = np.ones(T + 1)
mP2o1_mean = np.ones(T + 1)
mP0o2_std = np.zeros(T + 1)
mP1o2_std = np.zeros(T + 1)
mP1Co2_std = np.zeros(T + 1)
mP2o1_std = np.zeros(T + 1)
mPexp_std = np.zeros(T + 1)
mP0o2_final = np.zeros(size)
mP1o2_final = np.zeros(size)
mP1Co2_final = np.zeros(size)
mP2o1_final = np.zeros(size)
mPexp_final = np.zeros(size)

CP0o2_mean = np.zeros(T + 1)
CP1o2_mean = np.zeros(T + 1)
CP1Co2_mean = np.zeros(T + 1)
CPexp_mean = np.zeros(T + 1)
CP2o1_mean = np.zeros(T + 1)
CP0o2_std = np.zeros(T + 1)
CP1o2_std = np.zeros(T + 1)
CP1Co2_std = np.zeros(T + 1)
CP2o1_std = np.zeros(T + 1)
CPexp_std = np.zeros(T + 1)
CP0o2_final = np.zeros((size, size))
CP1o2_final = np.zeros((size, size))
CP1Co2_final = np.zeros((size, size))
CP2o1_final = np.zeros((size, size))
CPexp_final = np.zeros((size, size))

DP0o2_mean = np.zeros(T + 1)
DP1o2_mean = np.zeros(T + 1)
DP1Co2_mean = np.zeros(T + 1)
DPexp_mean = np.zeros(T + 1)
DP2o1_mean = np.zeros(T + 1)
DP0o2_std = np.zeros(T + 1)
DP1o2_std = np.zeros(T + 1)
DP1Co2_std = np.zeros(T + 1)
DP2o1_std = np.zeros(T + 1)
DPexp_std = np.zeros(T + 1)
DP0o2_final = np.zeros((size, size))
DP1o2_final = np.zeros((size, size))
DP1Co2_final = np.zeros((size, size))
DP2o1_final = np.zeros((size, size))
DPexp_final = np.zeros((size, size))


# Load data
filename = 'data/m-c-ts0-gamma1-' + str(gamma1) + '-gamma2-' + str(gamma2) + '-s-' + \
    str(size) + '-R-' + str(R) + '-beta-' + str(beta) + '.npz'
print(filename)
data = np.load(filename)
H = data['H']
J = data['J']
s0 = data['s0']


# Load statistical moments from data			
mPexp_mean[0] = 1
for t in range(T):
    print('Exp', str(t) + '/' + str(T))
    m_exp = data['m'][:, t]
    mPexp_mean[t + 1] = np.mean(m_exp)
    mPexp_std[t + 1] = np.std(m_exp)
    C_exp = data['C'][:, :, t]
    CPexp_mean[t + 1] = np.mean(C_exp[iu1])
    CPexp_std[t + 1] = np.std(C_exp[iu1])
    D_exp = data['D'][:, :, t]
    DPexp_mean[t + 1] = np.mean(D_exp)
    DPexp_std[t + 1] = np.std(D_exp)
    print(mPexp_mean[t + 1], CPexp_mean[t + 1], DPexp_mean[t + 1])
mPexp_final = m_exp
CPexp_final = C_exp
DPexp_final = D_exp


# Initialize kinetic Ising model
I = mf_ising(size)
I.H = H.copy()
I.J = J.copy()

# Run Plefka[t-1,t], order 2
I.initialize_state(s0)
for t in range(T):
    print('P0_o2', str(t) + '/' + str(T))
    I.update_P0_o2()
    CP0o2_mean[t + 1] = np.mean(I.C[iu1])
    CP0o2_std[t + 1] = np.std(I.C[iu1])
    mP0o2_mean[t + 1] = np.mean(I.m)
    mP0o2_std[t + 1] = np.std(I.m)
    DP0o2_mean[t + 1] = np.mean(I.D)
    DP0o2_std[t + 1] = np.std(I.D)
    EmP0o2[t + 1] = np.mean((I.m - data['m'][:, t])**2)
    ECP0o2[t + 1] = np.mean((I.C[iu1] - data['C'][:, :, t][iu1])**2)
    EDP0o2[t + 1] = np.mean((I.D - data['D'][:, :, t])**2)
    print(mP0o2_mean[t + 1], CP0o2_mean[t + 1], DP0o2_mean[t + 1])
mP0o2_final = I.m
CP0o2_final = I.C
DP0o2_final = I.D

# Run Plefka[t], order 2
I.initialize_state(s0)
for t in range(T):
    print('P1_o2', str(t) + '/' + str(T))
    I.update_P1_o2()
    CP1o2_mean[t + 1] = np.mean(I.C[iu1])
    CP1o2_std[t + 1] = np.std(I.C[iu1])
    mP1o2_mean[t + 1] = np.mean(I.m)
    mP1o2_std[t + 1] = np.std(I.m)
    DP1o2_mean[t + 1] = np.mean(I.D)
    DP1o2_std[t + 1] = np.std(I.D)
    EmP1o2[t + 1] = np.mean((I.m - data['m'][:, t])**2)
    ECP1o2[t + 1] = np.mean((I.C[iu1] - data['C'][:, :, t][iu1])**2)
    EDP1o2[t + 1] = np.mean((I.D - data['D'][:, :, t])**2)
    print(mP1o2_mean[t + 1], CP1o2_mean[t + 1], DP1o2_mean[t + 1])
mP1o2_final = I.m
CP1o2_final = I.C
DP1o2_final = I.D

# Run Plefka2[t], order 2
I.initialize_state(s0)
for t in range(T):
    print('P1C_o2', str(t) + '/' + str(T))
    I.update_P1C_o2()
    CP1Co2_mean[t + 1] = np.mean(I.C[iu1])
    CP1Co2_std[t + 1] = np.std(I.C[iu1])
    mP1Co2_mean[t + 1] = np.mean(I.m)
    mP1Co2_std[t + 1] = np.std(I.m)
    DP1Co2_mean[t + 1] = np.mean(I.D)
    DP1Co2_std[t + 1] = np.std(I.D)
    EmP1Co2[t + 1] = np.mean((I.m - data['m'][:, t])**2)
    ECP1Co2[t + 1] = np.mean((I.C[iu1] - data['C'][:, :, t][iu1])**2)
    EDP1Co2[t + 1] = np.mean((I.D - data['D'][:, :, t])**2)
    print(mP1Co2_mean[t + 1], CP1Co2_mean[t + 1], DP1Co2_mean[t + 1])
mP1Co2_final = I.m
CP1Co2_final = I.C
DP1Co2_final = I.D

# Run Plefka[t-1], order 1
I.initialize_state(s0)
for t in range(T):
    print('P2_o1', str(t) + '/' + str(T))
    I.update_P2_o1()
    CP2o1_mean[t + 1] = np.mean(I.C[iu1])
    CP2o1_std[t + 1] = np.std(I.C[iu1])
    mP2o1_mean[t + 1] = np.mean(I.m)
    mP2o1_std[t + 1] = np.std(I.m)
    DP2o1_mean[t + 1] = np.mean(I.D)
    DP2o1_std[t + 1] = np.std(I.D)
    EmP2o1[t + 1] = np.mean((I.m - data['m'][:, t])**2)
    ECP2o1[t + 1] = np.mean((I.C[iu1] - data['C'][:, :, t][iu1])**2)
    EDP2o1[t + 1] = np.mean((I.D - data['D'][:, :, t])**2)
mP2o1_final = I.m
CP2o1_final = I.C
DP2o1_final = I.D

# Save results to file

filename = 'img/compare-T_' + str(int(beta * 100)) + '_size_'+str(size)+'.npz'
np.savez_compressed(filename,
                    m_exp=data['m'][:, t], C_exp=data['C'][:,
                                                           :, t], D_exp=data['D'][:, :, t],
                    mP0o2_mean=mP0o2_mean, mP1o2_mean=mP1o2_mean, mP2o1_mean=mP2o1_mean, mP1Co2_mean=mP1Co2_mean,
                    CP0o2_mean=CP0o2_mean, CP1o2_mean=CP1o2_mean, CP2o1_mean=CP2o1_mean, CP1Co2_mean=CP1Co2_mean,
                    DP0o2_mean=DP0o2_mean, DP1o2_mean=DP1o2_mean, DP2o1_mean=DP2o1_mean, DP1Co2_mean=DP1Co2_mean,
                    mP0o2_std=mP0o2_std, mP1o2_std=mP1o2_std, mP2o1_std=mP2o1_std, mP1Co2_std=mP1Co2_std,
                    CP0o2_std=CP0o2_std, CP1o2_std=CP1o2_std, CP2o1_std=CP2o1_std, CP1Co2_std=CP1Co2_std,
                    DP0o2_std=DP0o2_std, DP1o2_std=DP1o2_std, DP2o1_std=DP2o1_std, DP1Co2_std=DP1Co2_std,
                    mPexp_mean=mPexp_mean, CPexp_mean=CPexp_mean, DPexp_mean=DPexp_mean,
                    mP0o2=mP0o2_final, mP1o2=mP1o2_final, mP2o1=mP2o1_final, mP1Co2=mP1Co2_final,
                    CP0o2=CP0o2_final, CP1o2=CP1o2_final, CP2o1=CP2o1_final, CP1Co2=CP1Co2_final,
                    DP0o2=DP0o2_final, DP1o2=DP1o2_final, DP2o1=DP2o1_final, DP1Co2=DP1Co2_final,
                    EmP0o2=EmP0o2, EmP1o2=EmP1o2, EmP2o1=EmP2o1, EmP1Co2=EmP1Co2,
                    ECP0o2=ECP0o2, ECP1o2=ECP1o2, ECP2o1=ECP2o1, ECP1Co2=ECP1Co2,
                    EDP0o2=EDP0o2, EDP1o2=EDP1o2, EDP2o1=EDP2o1, EDP1Co2=EDP1Co2)


# Plot results

steps = np.arange(T + 1)

cmap = cm.get_cmap('inferno_r')
colors = []
for i in range(4):
    colors += [cmap((i+0.5)/4)]
labels = [r'P[$t-1,t$]', r'P[$t$]', r'P[$t-1$]', r'P2[$t$]']


fig, ax = plt.subplots(1, 1, figsize=(5, 4))
plt.plot(steps, EmP0o2, 'b', label=labels[0])
plt.plot(steps, EmP1o2, 'g', label=labels[1])
plt.plot(steps, EmP2o1, 'm', label=labels[2])
plt.plot(steps, EmP1Co2, 'r', label=labels[3])
plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
plt.xlabel(r'$t$', fontsize=18)
plt.ylabel(r'$MSE[\textbf{m}_t]$', fontsize=18, rotation=0, labelpad=30)
plt.legend()
# plt.savefig('img/error_m-beta_' + str(int(beta * 10)) +
#            '.pdf', bbox_inches='tight')


fig, ax = plt.subplots(1, 1, figsize=(5, 4))
plt.plot(steps, ECP0o2, 'b', label=labels[0])
plt.plot(steps, ECP1o2, 'g', label=labels[1])
plt.plot(steps, ECP2o1, 'm', label=labels[2])
plt.plot(steps, ECP1Co2, 'r', label=labels[3])
plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
plt.xlabel(r'$t$', fontsize=18)
plt.ylabel(r'$MSE[\textbf{C}_t]$', fontsize=18, rotation=0, labelpad=30)
plt.legend()
# plt.savefig('img/error_C-beta_' + str(int(beta * 10)) +
#            '.pdf', bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
plt.plot(steps, EDP0o2, 'b', label=labels[0])
plt.plot(steps, EDP1o2, 'g', label=labels[1])
plt.plot(steps, EDP2o1, 'm', label=labels[2])
plt.plot(steps, EDP1Co2, 'r', label=r'P[D]')
plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
plt.xlabel(r'$t$', fontsize=18)
plt.ylabel(r'$MSE[\textbf{D}_t]$', fontsize=18, rotation=0, labelpad=30)
plt.legend()
# plt.savefig('img/error_D-beta_' + str(int(beta * 10)) +
#            '.pdf', bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
plt.plot(steps, mP0o2_mean, 'v', color=colors[0], ms=3, label=labels[0])
plt.plot(steps, mP1o2_mean, 's', color=colors[1], ms=3, label=labels[1])
plt.plot(steps, mP2o1_mean, 'd', color=colors[2], ms=3, label=labels[2])
plt.plot(steps, mP1Co2_mean, 'o', color=colors[3], ms=3, label=labels[3])
plt.plot(steps, mPexp_mean, 'k', label=r'$P$')
plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
plt.xlabel(r'$t$', fontsize=18)
plt.ylabel(r'$\langle m_{i,t} \rangle$', fontsize=18, rotation=0, labelpad=15)
plt.legend()
plt.savefig('img/evolution_m-beta_' + str(int(beta * 100)) +
            '.pdf', bbox_inches='tight')


fig, ax = plt.subplots(1, 1, figsize=(5, 4))
plt.plot(steps, CP0o2_mean, 'v', color=colors[0], ms=3, label=labels[0])
plt.plot(steps, CP1o2_mean, 's', color=colors[1], ms=3, label=labels[1])
plt.plot(steps, CP2o1_mean, 'd', color=colors[2], ms=3, label=labels[2])
plt.plot(steps, CP1Co2_mean, 'o', color=colors[3], ms=3, label=labels[3])
plt.plot(steps, CPexp_mean, 'k', label=r'$P$')
plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
plt.xlabel(r'$t$', fontsize=18)
plt.ylabel(r'$\langle C_{ik,t} \rangle$', fontsize=18, rotation=0, labelpad=15)
plt.legend(loc='lower right')
# plt.axis([0,T,0,1])
plt.savefig('img/evolution_C-beta_' + str(int(beta * 100)) +
            '.pdf', bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
plt.plot(steps, DP0o2_mean, 'v', color=colors[0], ms=3, label=labels[0])
plt.plot(steps, DP1o2_mean, 's', color=colors[1], ms=3, label=labels[1])
plt.plot(steps, DP2o1_mean, 'd', color=colors[2], ms=3, label=labels[2])
plt.plot(steps, DP1Co2_mean, 'o', color=colors[3], ms=3, label=labels[3])
plt.plot(steps, DPexp_mean, 'k', label=r'$P$')
plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
plt.xlabel(r'$t$', fontsize=18)
plt.ylabel(r'$\langle D_{il,t} \rangle$', fontsize=18, rotation=0, labelpad=15)
plt.legend(loc='lower right')
# plt.axis([0,T,0,1])
plt.savefig('img/evolution_D-beta_' + str(int(beta * 100)) +
            '.pdf', bbox_inches='tight')

plt.show()
