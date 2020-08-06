#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:45:08 2018

@author: maguilera
"""

from mf_ising import mf_ising
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import time


size = 512

R = 500000
mode = 'c'
gamma1 = 0.5
gamma2 = 0.1

T = 128
iu1 = np.triu_indices(size, 1)

if gamma1 == 0.5 and gamma2 == 0.1:
    beta0 = 1.1123
B = 21


betas = 1 + np.linspace(-1, 1, B) * 0.3
for ib in range(len(betas)):
    beta_ref = round(betas[ib], 3)
    beta = beta_ref * beta0

    EmP_t1_t = np.zeros(T + 1)
    EmP_t = np.zeros(T + 1)
    EmP2_t = np.zeros(T + 1)
    EmP_t1 = np.zeros(T + 1)

    ECP_t1_t = np.zeros(T + 1)
    ECP_t = np.zeros(T + 1)
    ECP2_t = np.zeros(T + 1)
    ECP_t1 = np.zeros(T + 1)

    EDP_t1_t = np.zeros(T + 1)
    EDP_t = np.zeros(T + 1)
    EDP2_t = np.zeros(T + 1)
    EDP_t1 = np.zeros(T + 1)


    mP_t1_t_mean = np.ones(T + 1)
    mP_t_mean = np.ones(T + 1)
    mP2_t_mean = np.ones(T + 1)
    mPexp_mean = np.ones(T + 1)
    mP_t1_mean = np.ones(T + 1)
    mP_t1_t_std = np.zeros(T + 1)
    mP_t_std = np.zeros(T + 1)
    mP2_t_std = np.zeros(T + 1)
    mP_t1_std = np.zeros(T + 1)
    mPexp_std = np.zeros(T + 1)
    mP_t1_t_final = np.zeros(size)
    mP_t_final = np.zeros(size)
    mP2_t_final = np.zeros(size)
    mP_t1_final = np.zeros(size)
    mPexp_final = np.zeros(size)

    CP_t1_t_mean = np.zeros(T + 1)
    CP_t_mean = np.zeros(T + 1)
    CP2_t_mean = np.zeros(T + 1)
    CPexp_mean = np.zeros(T + 1)
    CP_t1_mean = np.zeros(T + 1)
    CP_t1_t_std = np.zeros(T + 1)
    CP_t_std = np.zeros(T + 1)
    CP2_t_std = np.zeros(T + 1)
    CP_t1_std = np.zeros(T + 1)
    CPexp_std = np.zeros(T + 1)
    CP_t1_t_final = np.zeros((size, size))
    CP_t_final = np.zeros((size, size))
    CP2_t_final = np.zeros((size, size))
    CP_t1_final = np.zeros((size, size))
    CPexp_final = np.zeros((size, size))

    DP_t1_t_mean = np.zeros(T + 1)
    DP_t_mean = np.zeros(T + 1)
    DP2_t_mean = np.zeros(T + 1)
    DPexp_mean = np.zeros(T + 1)
    DP_t1_mean = np.zeros(T + 1)
    DP_t1_t_std = np.zeros(T + 1)
    DP_t_std = np.zeros(T + 1)
    DP2_t_std = np.zeros(T + 1)
    DP_t1_std = np.zeros(T + 1)
    DPexp_std = np.zeros(T + 1)
    DP_t1_t_final = np.zeros((size, size))
    DP_t_final = np.zeros((size, size))
    DP2_t_final = np.zeros((size, size))
    DP_t1_final = np.zeros((size, size))
    DPexp_final = np.zeros((size, size))


    # Load data
    filename = 'data/m-c-ts0-gamma1-' + str(gamma1) + '-gamma2-' + str(
        gamma2) + '-s-' + str(size) + '-R-' + str(R) + '-beta-' + str(beta_ref) + '.npz'
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
    time_start = time.perf_counter()
    I.initialize_state(s0)
    for t in range(T):
        print('beta',beta_ref,'P_t1_t_o2', str(t) + '/' + str(T))
        I.update_P_t1_t_o2()
        CP_t1_t_mean[t + 1] = np.mean(I.C[iu1])
        CP_t1_t_std[t + 1] = np.std(I.C[iu1])
        mP_t1_t_mean[t + 1] = np.mean(I.m)
        mP_t1_t_std[t + 1] = np.std(I.m)
        DP_t1_t_mean[t + 1] = np.mean(I.D)
        DP_t1_t_std[t + 1] = np.std(I.D)
        EmP_t1_t[t + 1] = np.mean((I.m - data['m'][:, t])**2)
        ECP_t1_t[t + 1] = np.mean((I.C[iu1] - data['C'][:, :, t][iu1])**2)
        EDP_t1_t[t + 1] = np.mean((I.D - data['D'][:, :, t])**2)
        print(mP_t1_t_mean[t + 1], CP_t1_t_mean[t + 1], DP_t1_t_mean[t + 1])
    mP_t1_t_final = I.m
    CP_t1_t_final = I.C
    DP_t1_t_final = I.D
    time_P_t1_t = time.perf_counter() - time_start

    # Run Plefka[t], order 2
    time_start = time.perf_counter()
    I.initialize_state(s0)
    for t in range(T):
        print('beta',beta_ref,'P_t_o2', str(t) + '/' + str(T))
        I.update_P_t_o2()
        CP_t_mean[t + 1] = np.mean(I.C[iu1])
        CP_t_std[t + 1] = np.std(I.C[iu1])
        mP_t_mean[t + 1] = np.mean(I.m)
        mP_t_std[t + 1] = np.std(I.m)
        DP_t_mean[t + 1] = np.mean(I.D)
        DP_t_std[t + 1] = np.std(I.D)
        EmP_t[t + 1] = np.mean((I.m - data['m'][:, t])**2)
        ECP_t[t + 1] = np.mean((I.C[iu1] - data['C'][:, :, t][iu1])**2)
        EDP_t[t + 1] = np.mean((I.D - data['D'][:, :, t])**2)
        print(mP_t_mean[t + 1], CP_t_mean[t + 1], DP_t_mean[t + 1])
    mP_t_final = I.m
    CP_t_final = I.C
    DP_t_final = I.D
    time_P_t = time.perf_counter() - time_start

    # Run Plefka2[t], order 2
    time_start = time.perf_counter()
    I.initialize_state(s0)
    for t in range(T):
        print('beta',beta_ref,'P2_t_o2', str(t) + '/' + str(T))
        I.update_P2_t_o2()
        CP2_t_mean[t + 1] = np.mean(I.C[iu1])
        CP2_t_std[t + 1] = np.std(I.C[iu1])
        mP2_t_mean[t + 1] = np.mean(I.m)
        mP2_t_std[t + 1] = np.std(I.m)
        DP2_t_mean[t + 1] = np.mean(I.D)
        DP2_t_std[t + 1] = np.std(I.D)
        EmP2_t[t + 1] = np.mean((I.m - data['m'][:, t])**2)
        ECP2_t[t + 1] = np.mean((I.C[iu1] - data['C'][:, :, t][iu1])**2)
        EDP2_t[t + 1] = np.mean((I.D - data['D'][:, :, t])**2)
        print(mP2_t_mean[t + 1], CP2_t_mean[t + 1], DP2_t_mean[t + 1])
    mP2_t_final = I.m
    CP2_t_final = I.C
    DP2_t_final = I.D
    time_P2_t = time.perf_counter() - time_start

    # Run Plefka[t-1], order 1
    time_start = time.perf_counter()
    I.initialize_state(s0)
    for t in range(T):
        print('beta',beta_ref,'P_t1_o1', str(t) + '/' + str(T))
        I.update_P_t1_o1()
        CP_t1_mean[t + 1] = np.mean(I.C[iu1])
        CP_t1_std[t + 1] = np.std(I.C[iu1])
        mP_t1_mean[t + 1] = np.mean(I.m)
        mP_t1_std[t + 1] = np.std(I.m)
        DP_t1_mean[t + 1] = np.mean(I.D)
        DP_t1_std[t + 1] = np.std(I.D)
        EmP_t1[t + 1] = np.mean((I.m - data['m'][:, t])**2)
        ECP_t1[t + 1] = np.mean((I.C[iu1] - data['C'][:, :, t][iu1])**2)
        EDP_t1[t + 1] = np.mean((I.D - data['D'][:, :, t])**2)
    mP_t1_final = I.m
    CP_t1_final = I.C
    DP_t1_final = I.D
    time_P_t1 = time.perf_counter() - time_start

    # Save results to file

    filename = 'img/compare-T_' + \
        str(int(beta_ref * 100)) + '_size_' + str(size) + '.npz'
    np.savez_compressed(filename,
                        m_exp=data['m'][:, t],
                        C_exp=data['C'][:, :, t],
                        D_exp=data['D'][:, :, t],
                        mP_t1_t_mean=mP_t1_t_mean,
                        mP_t_mean=mP_t_mean,
                        mP_t1_mean=mP_t1_mean,
                        mP2_t_mean=mP2_t_mean,
                        CP_t1_t_mean=CP_t1_t_mean,
                        CP_t_mean=CP_t_mean,
                        CP_t1_mean=CP_t1_mean,
                        CP2_t_mean=CP2_t_mean,
                        DP_t1_t_mean=DP_t1_t_mean,
                        DP_t_mean=DP_t_mean,
                        DP_t1_mean=DP_t1_mean,
                        DP2_t_mean=DP2_t_mean,
                        mP_t1_t_std=mP_t1_t_std,
                        mP_t_std=mP_t_std,
                        mP_t1_std=mP_t1_std,
                        mP2_t_std=mP2_t_std,
                        CP_t1_t_std=CP_t1_t_std,
                        CP_t_std=CP_t_std,
                        CP_t1_std=CP_t1_std,
                        CP2_t_std=CP2_t_std,
                        DP_t1_t_std=DP_t1_t_std,
                        DP_t_std=DP_t_std,
                        DP_t1_std=DP_t1_std,
                        DP2_t_std=DP2_t_std,
                        mPexp_mean=mPexp_mean,
                        CPexp_mean=CPexp_mean,
                        DPexp_mean=DPexp_mean,
                        mP_t1_t=mP_t1_t_final,
                        mP_t=mP_t_final,
                        mP_t1=mP_t1_final,
                        mP2_t=mP2_t_final,
                        CP_t1_t=CP_t1_t_final,
                        CP_t=CP_t_final,
                        CP_t1=CP_t1_final,
                        CP2_t=CP2_t_final,
                        DP_t1_t=DP_t1_t_final,
                        DP_t=DP_t_final,
                        DP_t1=DP_t1_final,
                        DP2_t=DP2_t_final,
                        EmP_t1_t=EmP_t1_t,
                        EmP_t=EmP_t,
                        EmP_t1=EmP_t1,
                        EmP2_t=EmP2_t,
                        ECP_t1_t=ECP_t1_t,
                        ECP_t=ECP_t,
                        ECP_t1=ECP_t1,
                        ECP2_t=ECP2_t,
                        EDP_t1_t=EDP_t1_t,
                        EDP_t=EDP_t,
                        EDP_t1=EDP_t1,
                        EDP2_t=EDP2,
                        time_P_t1_t=time_P_t1_t,
                        time_P_t=time_P_t,
                        time_P_t1=time_P_t1,
                        time_P2_t=time_P2_t)

    # Plot results

    steps = np.arange(T + 1)

    cmap = cm.get_cmap('inferno_r')
    colors = []
    for i in range(4):
        colors += [cmap((i + 0.5) / 4)]
    labels = [r'P[$t-1,t$]', r'P[$t$]', r'P[$t-1$]', r'P2[$t$]']


    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    plt.plot(steps, EmP_t1_t, 'b', label=labels[0])
    plt.plot(steps, EmP_t, 'g', label=labels[1])
    plt.plot(steps, EmP_t1, 'm', label=labels[2])
    plt.plot(steps, EmP2_t, 'r', label=labels[3])
    plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$MSE[\textbf{m}_t]$', fontsize=18, rotation=0, labelpad=30)
    plt.legend()
    # plt.savefig('img/error_m-beta_' + str(int(beta * 10)) +
    #            '.pdf', bbox_inches='tight')


    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    plt.plot(steps, ECP_t1_t, 'b', label=labels[0])
    plt.plot(steps, ECP_t, 'g', label=labels[1])
    plt.plot(steps, ECP_t1, 'm', label=labels[2])
    plt.plot(steps, ECP2_t, 'r', label=labels[3])
    plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$MSE[\textbf{C}_t]$', fontsize=18, rotation=0, labelpad=30)
    plt.legend()
    # plt.savefig('img/error_C-beta_' + str(int(beta * 10)) +
    #            '.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    plt.plot(steps, EDP_t1_t, 'b', label=labels[0])
    plt.plot(steps, EDP_t, 'g', label=labels[1])
    plt.plot(steps, EDP_t1, 'm', label=labels[2])
    plt.plot(steps, EDP2_t, 'r', label=r'P[D]')
    plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$MSE[\textbf{D}_t]$', fontsize=18, rotation=0, labelpad=30)
    plt.legend()
    # plt.savefig('img/error_D-beta_' + str(int(beta * 10)) +
    #            '.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    plt.plot(steps, mP_t1_t_mean, 'v', color=colors[0], ms=3, label=labels[0])
    plt.plot(steps, mP_t_mean, 's', color=colors[1], ms=3, label=labels[1])
    plt.plot(steps, mP_t1_mean, 'd', color=colors[2], ms=3, label=labels[2])
    plt.plot(steps, mP2_t_mean, 'o', color=colors[3], ms=3, label=labels[3])
    plt.plot(steps, mPexp_mean, 'k', label=r'$P$')
    plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$\langle m_{i,t} \rangle$', fontsize=18, rotation=0, labelpad=15)
    plt.legend()
    plt.savefig('img/evolution_m-beta_' + str(int(beta * 100)) +
                '.pdf', bbox_inches='tight')


    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    plt.plot(steps, CP_t1_t_mean, 'v', color=colors[0], ms=3, label=labels[0])
    plt.plot(steps, CP_t_mean, 's', color=colors[1], ms=3, label=labels[1])
    plt.plot(steps, CP_t1_mean, 'd', color=colors[2], ms=3, label=labels[2])
    plt.plot(steps, CP2_t_mean, 'o', color=colors[3], ms=3, label=labels[3])
    plt.plot(steps, CPexp_mean, 'k', label=r'$P$')
    plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$\langle C_{ik,t} \rangle$', fontsize=18, rotation=0, labelpad=15)
    plt.legend(loc='lower right')
    # plt.axis([0,T,0,1])
    plt.savefig('img/evolution_C-beta_' + str(int(beta * 100)) +
                '.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    plt.plot(steps, DP_t1_t_mean, 'v', color=colors[0], ms=3, label=labels[0])
    plt.plot(steps, DP_t_mean, 's', color=colors[1], ms=3, label=labels[1])
    plt.plot(steps, DP_t1_mean, 'd', color=colors[2], ms=3, label=labels[2])
    plt.plot(steps, DP2_t_mean, 'o', color=colors[3], ms=3, label=labels[3])
    plt.plot(steps, DPexp_mean, 'k', label=r'$P$')
    plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$\langle D_{il,t} \rangle$', fontsize=18, rotation=0, labelpad=15)
    plt.legend(loc='lower right')
    # plt.axis([0,T,0,1])
    plt.savefig('img/evolution_D-beta_' + str(int(beta * 100)) +
                '.pdf', bbox_inches='tight')
    
    #Close figures
    plt.close('all') 
