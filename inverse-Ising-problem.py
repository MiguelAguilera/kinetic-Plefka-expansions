#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:45:08 2018

@author: maguilera
"""

from mf_ising import mf_ising
from ising_functions import update_m_P_t1_t_o2, update_D_P_t1_t_o2, update_m_P_t_o2, update_D_P_t_o2
from ising_functions import update_m_P_t1_o1, update_D_P_t1_o1, update_D_P2_t_o2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import time

plt.rc('text', usetex=True)
font = {'size': 15}
plt.rc('font', **font)
plt.rc('legend', **{'fontsize': 12})


size = 512
R = 300000
mode = 'c'
gamma1 = 0.5
gamma2 = 0.1

T = 128 - 1


cmap = cm.get_cmap('inferno_r')
colors = []
for i in range(4):
    colors += [cmap((i + 0.5) / 4)]


if gamma1 == 0.5 and gamma2 == 0.1:
    beta0 = 1.1123
B = 21
betas = 1 + np.linspace(-1, 1, B) * 0.3
betas=np.array([1.0])
#betas = betas[2:]
#betas = betas[betas > 1.1]
print(betas)

eta = 1*0.2
error_ref = 1E-6
max_rep = 1000
max_rep_min = 5

etaH = eta
#etaJ=eta/size*5
etaJ = eta / size**0.5
st = 32
timesteps = np.arange(st // 2, T - 1, st)
timesteps=np.array([128-2])
#timesteps = np.array([10,20,30])
print(timesteps)
Nsample = len(timesteps)
    
for ib in range(len(betas)):
    beta_ref = round(betas[ib], 3)
    beta = beta_ref * beta0
    T = 128
    # T=2

    filename = 'data/m-c-ts0-gamma1-' + str(gamma1) + '-gamma2-' + str(
        gamma2) + '-s-' + str(size) + '-R-' + str(R) + '-beta-' + str(beta_ref) + '.npz'
    data = np.load(filename)
    H = data['H']
    J = data['J']

    iu1 = np.triu_indices(size, 1)
    s0 = np.array(data['s0'])




    # Run Plefka[t-1,t], order 2
    time_start = time.perf_counter()
    I = mf_ising(size)
    error = 1
    min_error = error

    for c, t in enumerate(timesteps):
        I.H += np.arctanh(data['m'][:, t]) / Nsample
    
    rep = 0
    rep_min = 0

    while error > error_ref:
        break
        DH = np.zeros(size)
        DJ = np.zeros((size, size))
        DH_mean = np.zeros(size)
        DJ_mean = np.zeros((size, size))
        for c, t in enumerate(timesteps):
            I.m_p = data['m'][:, t]
            I.m = update_m_P_t1_t_o2(I.H, I.J, I.m_p)
            I.D = update_D_P_t1_t_o2(I.H, I.J, I.m, I.m_p)
            DJ =  (data['D'][:, :, t + 1] + np.einsum('i,j->ij',data['m'][:, t + 1],I.m_p,optimize=True)) - (I.D + np.einsum('i,j->ij',I.m,I.m_p,optimize=True))
            DH = data['m'][:, t + 1] - I.m
            DJ_mean += DJ / Nsample
            DH_mean += DH / Nsample

        error = max(np.max(np.abs(DH_mean)), np.max(np.abs(DJ_mean)))
        error2 = np.mean(DJ_mean**2)
        if error2 < min_error:
            HP_t1_t = I.H.copy()
            JP_t1_t = I.J.copy()
            min_error = error2
            rep_min = 0
        I.J += etaJ * DJ_mean
        I.H += etaH * DH_mean
        print('P_t1_t_o2', beta_ref, rep, np.max(np.abs(DH_mean)),np.max(np.abs(DJ_mean)), np.max(np.abs(H - I.H)), np.max(np.abs(J - I.J)), np.mean((H - I.H)**2), np.mean((J - I.J)**2))
        rep += 1
        rep_min += 1
        if error < error_ref or rep > max_rep:#or (rep > (max_rep / 10) and rep_min > max_rep_min):
            print(error < error_ref, rep > max_rep, rep_min > max_rep_min)
            break
    time_P_t1_t = time.perf_counter() - time_start
    
    
    # Run Plefka[t], order 2
    time_start = time.perf_counter()
    I = mf_ising(size)
    error = 1
    min_error = error
    for c, t in enumerate(timesteps):
        I.H += np.arctanh(data['m'][:, t]) / Nsample
    I.H = H.copy()
    I.J = J.copy()
    rep = 0
    rep_min = 0
    while error > error_ref:
        break
        DH = np.zeros(size)
        DJ = np.zeros((size, size))
        DH_mean = np.zeros(size)
        DJ_mean = np.zeros((size, size))
        for c, t in enumerate(timesteps):
            I.m_p = data['m'][:, t]
            I.C_p = data['C'][:, :, t]
            I.m = update_m_P_t_o2(I.H, I.J, I.m_p, I.C_p)
            I.D = update_D_P_t_o2(I.H, I.J, I.m, I.m_p, I.C_p)
            DJ =  (data['D'][:, :, t + 1] + np.einsum('i,j->ij',data['m'][:, t + 1],I.m_p,optimize=True)) - (I.D + np.einsum('i,j->ij',I.m,I.m_p,optimize=True))
            DH = data['m'][:, t + 1] - I.m
            DJ_mean += DJ / Nsample
            DH_mean += DH / Nsample

#        print(np.mean(DH_mean**2),np.mean(DJ_mean**2))
#        plt.figure()
#        plt.plot([np.min(data['m'][:, t + 1]),np.max(data['m'][:, t + 1])],[np.min(data['m'][:, t + 1]),np.max(data['m'][:, t + 1])],'k')
#        plt.plot(data['m'][:, t + 1], I.m,'.')
#        plt.figure()
#        plt.plot([np.min(data['D'][:, :, t + 1]),np.max(data['D'][:, :, t + 1])],[np.min(data['D'][:, :, t + 1]),np.max(data['D'][:, :, t + 1])],'k')
#        plt.plot(data['D'][:, :, t + 1].flatten(),I.D.flatten(),'.')
#        break

        error = max(np.max(np.abs(DH_mean)), np.max(np.abs(DJ_mean)))
        error2 = np.mean(DJ_mean**2)
        if error2 < min_error:
            HP_t = I.H.copy()
            JP_t = I.J.copy()
            min_error = error2
            rep_min = 0
        I.J += etaJ * DJ_mean
        I.H += etaH * DH_mean
        print('P_t_o2', beta_ref, rep, np.max(np.abs(DH_mean)), np.max(np.abs(DJ_mean)), np.max(np.abs(H - I.H)), np.max(np.abs(J - I.J)), np.mean((H - I.H)**2), np.mean((J - I.J)**2))
        rep += 1
        rep_min += 1
        if error < error_ref or rep > max_rep:#or (rep > (max_rep / 10) and rep_min > max_rep_min):
            print(error < error_ref, rep > max_rep, rep_min > max_rep_min)
            break
    time_P_t = time.perf_counter() - time_start


    # Run Plefka2[t], order 2
    time_start = time.perf_counter()
    I = mf_ising(size)
    error = 1
    min_error = error
    for c, t in enumerate(timesteps):
        I.H += np.arctanh(data['m'][:, t]) / Nsample
    I.H = H.copy()
    I.J = J.copy()
    rep = 0
    rep_min = 0
    while error > error_ref:
        DH_mean = np.zeros(size)
        DJ_mean = np.zeros((size, size))
        for c, t in enumerate(timesteps):
            I.m_p = data['m'][:, t]
            I.C_p = data['C'][:, :, t]
            I.D_p = data['D'][:, :, t]
            I.m, _, I.D = update_D_P2_t_o2(I.H, I.J, I.m_p, I.C_p, I.D_p)
            DJ =  (data['D'][:, :, t + 1] + np.einsum('i,j->ij',data['m'][:, t + 1],I.m_p,optimize=True)) - (I.D + np.einsum('i,j->ij',I.m,I.m_p,optimize=True))
            DH = data['m'][:, t + 1] - I.m
            DJ_mean += DJ / Nsample
            DH_mean += DH / Nsample

#        print(np.mean(DH_mean**2),np.mean(DJ_mean**2))
#        plt.figure()
#        plt.plot([np.min(data['m'][:, t + 1]),np.max(data['m'][:, t + 1])],[np.min(data['m'][:, t + 1]),np.max(data['m'][:, t + 1])],'k')
#        plt.plot(data['m'][:, t + 1], I.m,'.')
#        plt.figure()
#        plt.plot([np.min(data['D'][:, :, t + 1]),np.max(data['D'][:, :, t + 1])],[np.min(data['D'][:, :, t + 1]),np.max(data['D'][:, :, t + 1])],'k')
#        plt.plot(data['D'][:, :, t + 1].flatten(),I.D.flatten(),'.')
#        plt.show()
        
        error = max(np.max(np.abs(DH_mean)), np.max(np.abs(DJ_mean)))
        error2 = np.mean(DJ_mean**2)
        if error2 < min_error:
            HP2_t = I.H.copy()
            JP2_t = I.J.copy()
            min_error = error2
            rep_min = 0
        I.J += etaJ * DJ_mean
        I.H += etaH * DH_mean
        print('P2_t_o2', beta_ref, rep, np.max(np.abs(DH_mean)), np.max(np.abs(DJ_mean)), np.max(np.abs(H - I.H)), np.max(np.abs(J - I.J)), np.mean((H - I.H)**2), np.mean((J - I.J)**2))
        rep += 1
        rep_min += 1
        if error < error_ref or rep > max_rep:#or (rep > (max_rep / 10) and rep_min > max_rep_min):
            print(error < error_ref, rep > max_rep, rep_min > max_rep_min)
            break
    time_P2_t = time.perf_counter() - time_start

    # Run Plefka[t-1], order 1
    time_start = time.perf_counter()
    I = mf_ising(size)
    error = 1
    min_error = error
    for c, t in enumerate(timesteps):
        I.H += np.arctanh(data['m'][:, t]) / Nsample

    rep = 0
    rep_min = 0
    while error > error_ref:
        DH = np.zeros(size)
        DJ = np.zeros((size, size))
        DH_mean = np.zeros(size)
        DJ_mean = np.zeros((size, size))
        for c, t in enumerate(timesteps):
            I.m_p = data['m'][:, t]
            I.C_p = data['C'][:, :, t]
            I.m = update_m_P_t1_o1(I.H, I.J,)
            I.D = update_D_P_t1_o1(I.H, I.J, I.m_p, I.C_p)
            DJ =  (data['D'][:, :, t + 1] + np.einsum('i,j->ij',data['m'][:, t + 1],I.m_p,optimize=True)) - (I.D + np.einsum('i,j->ij',I.m,I.m_p,optimize=True))
            DH = data['m'][:, t + 1] - I.m
            DJ_mean += DJ / Nsample
            DH_mean += DH / Nsample

        error = max(np.max(np.abs(DH_mean)), np.max(np.abs(DJ_mean)))
        error2 = np.mean(DJ_mean**2)
        if error2 < min_error:
            HP_t1 = I.H.copy()
            JP_t1 = I.J.copy()
            min_error = error2
            rep_min = 0
        I.J += etaJ * DJ_mean
        I.H += etaH * DH_mean

        print('P_t1_o1', beta_ref, rep, np.max(np.abs(DH_mean)), np.max(np.abs(DJ_mean)), np.max(np.abs(H - I.H)), np.max(np.abs(J - I.J)), np.mean((H - I.H)**2), np.mean((J - I.J)**2))
        rep += 1
        rep_min += 1
        if error < error_ref or rep > max_rep:#or (rep > (max_rep / 10) and rep_min > max_rep_min):
            print(error < error_ref, rep > max_rep, rep_min > max_rep_min)
            break
    time_P_t1 = time.perf_counter() - time_start

    filename = 'data/results/inverse_' + str(int(beta_ref * 100)) +'_R_' + str(R) +'.npz'
    np.savez_compressed(filename,
                        H=H, J=J,
                        HP_t1_t=HP_t1_t, HP_t=HP_t, HP_t1=HP_t1, HP2_t=HP2_t,
                        JP_t1_t=JP_t1_t, JP_t=JP_t, JP_t1=JP_t1, JP2_t=JP2_t,
                        time_P_t1_t=time_P_t1_t,
                        time_P_t=time_P_t,
                        time_P_t1=time_P_t1,
                        time_P2_t=time_P2_t)
