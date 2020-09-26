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

R = 1000000
gamma1 = 0.5
gamma2 = 0.1

T = 128
iu1 = np.triu_indices(size, 1)

B = 61


betas = 1 + np.linspace(-1, 1, B) * 0.3
#betas=np.array([1.0])
#betas=np.array([0.97])

filename1 = 'data/data-gamma1-' + str(gamma1) + '-gamma2-' + str(
        gamma2) + '-s-' + str(size) + '-R-' + str(R) + '-beta-1.0.npz'
data1 = np.load(filename1)
s0 = data1['s0']
del data1

modes=['d','r']        # Direct and reconstruction modes
mode = modes[1]
for ib in range(len(betas)):
    beta_ref = round(betas[ib], 3)


    # Load data
    
    filename = 'data/results/inverse_100_R_' + str(R) +'.npz'
    print(beta_ref,mode)
    
    data = np.load(filename)
    HP_t1_t = data['HP_t1_t']*beta_ref
    JP_t1_t = data['JP_t1_t']*beta_ref
    HP_t = data['HP_t']*beta_ref
    JP_t = data['JP_t']*beta_ref
    HP_t1 = data['HP_t1']*beta_ref
    JP_t1 = data['JP_t1']*beta_ref
    HP2_t = data['HP2_t']*beta_ref
    JP2_t = data['JP2_t']*beta_ref
    

    J=data['J']*beta_ref
    H=data['H']*beta_ref
    del data
#    plt.figure()
#    plt.plot(H,HP2_t,'*')
#    plt.plot([np.min(H),np.max(H)],[np.min(H),np.max(H)],'k')
#    plt.figure()
#    plt.plot(J.flatten(),JP2_t.flatten(),'.')
#    plt.plot([np.min(J),np.max(J)],[np.min(J),np.max(J)],'k')
#    
#    plt.figure()
#    plt.plot(H,HP_t,'*')
#    plt.plot([np.min(H),np.max(H)],[np.min(H),np.max(H)],'k')
#    plt.figure()
#    plt.plot(J.flatten(),JP_t.flatten(),'.')
#    plt.plot([np.min(J),np.max(J)],[np.min(J),np.max(J)],'k')

#    print('P',np.mean((H-HP_t)**2),np.mean((J-JP_t)**2))
#    print('P2',np.mean((H-HP2_t)**2),np.mean((J-JP2_t)**2))
#    plt.show()

    # Run Plefka[t-1,t], order 2
    I = mf_ising(size)
    if mode == 'd':
        I.H = H.copy()
        I.J = J.copy()
    elif mode =='r':
        I.H = HP_t1_t.copy()
        I.J = JP_t1_t.copy()
    I.initialize_state(s0)
    for t in range(T):
        print('beta',beta_ref,'P_t1_t_o2', str(t) + '/' + str(T),np.mean(I.m),np.mean(I.C[iu1]),np.mean(I.D))
        I.update_P_t1_t_o2()
    mP_t1_t_final = I.m
    CP_t1_t_final = I.C
    DP_t1_t_final = I.D

    # Run Plefka[t], order 2
    I = mf_ising(size)
    if mode == 'd':
        I.H = H.copy()
        I.J = J.copy()
    elif mode =='r':
        I.H = HP_t.copy()
        I.J = JP_t.copy()
    I.initialize_state(s0)
    for t in range(T):
        print('beta',beta_ref,'P_t_o2', str(t) + '/' + str(T),np.mean(I.m),np.mean(I.C[iu1]),np.mean(I.D))
        I.update_P_t_o2()
    mP_t_final = I.m
    CP_t_final = I.C
    DP_t_final = I.D

    # Run Plefka2[t], order 2
    I = mf_ising(size)
    if mode == 'd':
        I.H = H.copy()
        I.J = J.copy()
    elif mode =='r':
        I.H = HP2_t.copy()
        I.J = JP2_t.copy()
#    I.H = H.copy()
#    I.J = J.copy()
    I.initialize_state(s0)
    for t in range(T):
        print('beta',beta_ref,'P2_t_o2', str(t) + '/' + str(T),np.mean(I.m),np.mean(I.C[iu1]),np.mean(I.D))
        I.update_P2_t_o2()
    mP2_t_final = I.m
    CP2_t_final = I.C
    DP2_t_final = I.D

    # Run Plefka[t-1], order 1
    I = mf_ising(size)
    if mode == 'd':
        I.H = H.copy()
        I.J = J.copy()
    elif mode =='r':
        I.H = HP_t1.copy()
        I.J = JP_t1.copy()
    I.initialize_state(s0)
    for t in range(T):
        print('beta',beta_ref,'P_t1_o1', str(t) + '/' + str(T),np.mean(I.m),np.mean(I.C[iu1]),np.mean(I.D))
        I.update_P_t1_o1()
    mP_t1_final = I.m
    CP_t1_final = I.C
    DP_t1_final = I.D

    # Save results to file

    filename = 'data/results/transition_'+mode+'_' + str(int(beta_ref * 100)) +'_R_' + str(R) +'.npz'
    np.savez_compressed(filename,
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
                        DP2_t=DP2_t_final)

#    # Plot results

#    steps = np.arange(T + 1)

#    cmap = cm.get_cmap('inferno_r')
#    colors = []
#    for i in range(4):
#        colors += [cmap((i + 0.5) / 4)]
#    labels = [r'P[$t-1,t$]', r'P[$t$]', r'P[$t-1$]', r'P2[$t$]']


#    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
#    plt.plot(steps, EmP_t1_t, 'b', label=labels[0])
#    plt.plot(steps, EmP_t, 'g', label=labels[1])
#    plt.plot(steps, EmP_t1, 'm', label=labels[2])
#    plt.plot(steps, EmP2_t, 'r', label=labels[3])
#    plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
#    plt.xlabel(r'$t$', fontsize=18)
#    plt.ylabel(r'$MSE[\textbf{m}_t]$', fontsize=18, rotation=0, labelpad=30)
#    plt.legend()
#    # plt.savefig('img/error_m-beta_' + str(int(beta * 10)) +
#    #            '.pdf', bbox_inches='tight')


#    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
#    plt.plot(steps, ECP_t1_t, 'b', label=labels[0])
#    plt.plot(steps, ECP_t, 'g', label=labels[1])
#    plt.plot(steps, ECP_t1, 'm', label=labels[2])
#    plt.plot(steps, ECP2_t, 'r', label=labels[3])
#    plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
#    plt.xlabel(r'$t$', fontsize=18)
#    plt.ylabel(r'$MSE[\textbf{C}_t]$', fontsize=18, rotation=0, labelpad=30)
#    plt.legend()
#    # plt.savefig('img/error_C-beta_' + str(int(beta * 10)) +
#    #            '.pdf', bbox_inches='tight')

#    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
#    plt.plot(steps, EDP_t1_t, 'b', label=labels[0])
#    plt.plot(steps, EDP_t, 'g', label=labels[1])
#    plt.plot(steps, EDP_t1, 'm', label=labels[2])
#    plt.plot(steps, EDP2_t, 'r', label=r'P[D]')
#    plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
#    plt.xlabel(r'$t$', fontsize=18)
#    plt.ylabel(r'$MSE[\textbf{D}_t]$', fontsize=18, rotation=0, labelpad=30)
#    plt.legend()
#    # plt.savefig('img/error_D-beta_' + str(int(beta * 10)) +
#    #            '.pdf', bbox_inches='tight')

#    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
#    plt.plot(steps, mP_t1_t_mean, 'v', color=colors[0], ms=3, label=labels[0])
#    plt.plot(steps, mP_t_mean, 's', color=colors[1], ms=3, label=labels[1])
#    plt.plot(steps, mP_t1_mean, 'd', color=colors[2], ms=3, label=labels[2])
#    plt.plot(steps, mP2_t_mean, 'o', color=colors[3], ms=3, label=labels[3])
#    plt.plot(steps, mPexp_mean, 'k', label=r'$P$')
#    plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
#    plt.xlabel(r'$t$', fontsize=18)
#    plt.ylabel(r'$\langle m_{i,t} \rangle$', fontsize=18, rotation=0, labelpad=15)
#    plt.legend()
##    plt.savefig('img/evolution_m-beta_' + str(int(beta * 100)) +
##                '.pdf', bbox_inches='tight')


#    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
#    plt.plot(steps, CP_t1_t_mean, 'v', color=colors[0], ms=3, label=labels[0])
#    plt.plot(steps, CP_t_mean, 's', color=colors[1], ms=3, label=labels[1])
#    plt.plot(steps, CP_t1_mean, 'd', color=colors[2], ms=3, label=labels[2])
#    plt.plot(steps, CP2_t_mean, 'o', color=colors[3], ms=3, label=labels[3])
#    plt.plot(steps, CPexp_mean, 'k', label=r'$P$')
#    plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
#    plt.xlabel(r'$t$', fontsize=18)
#    plt.ylabel(r'$\langle C_{ik,t} \rangle$', fontsize=18, rotation=0, labelpad=15)
#    plt.legend(loc='lower right')
#    # plt.axis([0,T,0,1])
##    plt.savefig('img/evolution_C-beta_' + str(int(beta * 100)) +
##                '.pdf', bbox_inches='tight')

#    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
#    plt.plot(steps, DP_t1_t_mean, 'v', color=colors[0], ms=3, label=labels[0])
#    plt.plot(steps, DP_t_mean, 's', color=colors[1], ms=3, label=labels[1])
#    plt.plot(steps, DP_t1_mean, 'd', color=colors[2], ms=3, label=labels[2])
#    plt.plot(steps, DP2_t_mean, 'o', color=colors[3], ms=3, label=labels[3])
#    plt.plot(steps, DPexp_mean, 'k', label=r'$P$')
#    plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
#    plt.xlabel(r'$t$', fontsize=18)
#    plt.ylabel(r'$\langle D_{il,t} \rangle$', fontsize=18, rotation=0, labelpad=15)
#    plt.legend(loc='lower right')
#    # plt.axis([0,T,0,1])
##    plt.savefig('img/evolution_D-beta_' + str(int(beta * 100)) +
##                '.pdf', bbox_inches='tight')
#    
#    #Close figures
#    plt.close('all') 
