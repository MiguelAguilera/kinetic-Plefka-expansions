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
from sys import argv

if len(argv) < 2:
    print("Usage: " + argv[0] + " <beta_ref>" )
    exit(1)

beta_ref = float(argv[1])        # beta / beta_c value

plt.rc('text', usetex=True)
font = {'size': 15}
plt.rc('font', **font)
plt.rc('legend', **{'fontsize': 12})


size = 512
R = 500000
Rv = 100000
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

eta = 1
#eta = 0.02
#eta = 0.001
validation_rep = 10
max_rep = 1000

etaH = eta
#etaJ=eta/size*5
etaJ = 0.2 * eta / size**0.5
st = 16
timesteps = np.arange(st // 2, T - 2, st)
#timesteps=np.array([128-2])
#timesteps = np.array([16])
#timesteps = np.array([120,122,124])
#timesteps = np.array([114,116,118,120,122,124])

print(timesteps)
Nsample = len(timesteps)
    
#for ib in range(len(betas)):


#beta_ref = round(betas[ib], 3)


beta = beta_ref * beta0
T = 128
# T=2

filename = 'data/m-c-ts0-gamma1-' + str(gamma1) + '-gamma2-' + str(
    gamma2) + '-s-' + str(size) + '-R-' + str(R) + '-beta-' + str(beta_ref) + '.npz'
data = np.load(filename)
H = data['H']
J = data['J']
m_exp=data['m']
C_exp=data['C']
D_exp=data['D']
s0 = np.array(data['s0'])
del data

filename_v = 'data/m-c-ts0-gamma1-' + str(gamma1) + '-gamma2-' + str(
    gamma2) + '-s-' + str(size) + '-R-' + str(Rv) + '-beta-' + str(beta_ref) + '.npz'
data_v = np.load(filename_v)
m_exp_v=data_v['m']
C_exp_v=data_v['C']
D_exp_v=data_v['D']
del data_v
iu1 = np.triu_indices(size, 1)





# Run Plefka[t-1,t], order 2
time_start = time.perf_counter()
I = mf_ising(size)
error_J = 1
error_H = 1
for c, t in enumerate(timesteps):
    I.H += np.arctanh(m_exp[:, t]) / Nsample

rep = 0

cond=True
while cond:
    DH = np.zeros(size)
    DJ = np.zeros((size, size))
    DH_mean = np.zeros(size)
    DJ_mean = np.zeros((size, size))
    for c, t in enumerate(timesteps):
        I.m_p = m_exp[:, t]
        I.m = update_m_P_t1_t_o2(I.H, I.J, I.m_p)
        I.D = update_D_P_t1_t_o2(I.H, I.J, I.m, I.m_p)
        DJ =  (D_exp[:, :, t + 1] + np.einsum('i,j->ij',m_exp[:, t + 1],I.m_p,optimize=True)) - (I.D + np.einsum('i,j->ij',I.m,I.m_p,optimize=True))
        DH = m_exp[:, t + 1] - I.m
        DJ_mean += DJ / Nsample
        DH_mean += DH / Nsample

    I.J += etaJ * DJ_mean
    I.H += etaH * DH_mean
    print('P_t1_t_o2', beta_ref, rep, np.mean(DH_mean**2),np.mean(DJ_mean**2), np.mean((H - I.H)**2), np.mean((J - I.J)**2))
    
    # Every few iterations, check errors in validation data.
    # If error does not decrease, early stop gradient descent
    if rep%validation_rep==0:
        error_H_prev = error_H
        error_J_prev = error_J
        DH_mean_v = np.zeros(size)
        DJ_mean_v = np.zeros((size, size))
        for c, t in enumerate(timesteps):
            I.m_p = m_exp_v[:, t]
            I.m = update_m_P_t1_t_o2(I.H, I.J, I.m_p)
            I.D = update_D_P_t1_t_o2(I.H, I.J, I.m, I.m_p)
            DJ =  (D_exp_v[:, :, t+1] + np.einsum('i,j->ij',m_exp_v[:, t+1],I.m_p,optimize=True)) - (I.D + np.einsum('i,j->ij',I.m,I.m_p,optimize=True))
            DH = m_exp_v[:, t+1] - I.m
            DH_mean_v += DH / Nsample
            DJ_mean_v += DJ / Nsample
        error_H = np.mean(DH_mean_v**2)
        error_J = np.mean(DJ_mean_v**2)
        
        # Stop condition
        print('P_t1_t_o2', 'Stop condition', np.mean(DH_mean**2),np.mean(DJ_mean**2),np.mean(DH_mean_v**2),np.mean(DJ_mean_v**2))
        if error_H > error_H_prev or error_J > error_J_prev or rep > max_rep:
            cond=False
        else:
            HP_t1_t = I.H.copy()
            JP_t1_t = I.J.copy()
    rep += 1
time_P_t1_t = time.perf_counter() - time_start


# Run Plefka[t], order 2
time_start = time.perf_counter()
I = mf_ising(size)
error_J = 1
error_H = 1
for c, t in enumerate(timesteps):
    I.H += np.arctanh(m_exp[:, t]) / Nsample
#I.H = H.copy()
#I.J = J.copy()
rep = 0
cond=True
while cond:
    DH = np.zeros(size)
    DJ = np.zeros((size, size))
    DH_mean = np.zeros(size)
    DJ_mean = np.zeros((size, size))
    for c, t in enumerate(timesteps):
        I.m_p = m_exp[:, t]
        I.C_p = C_exp[:, :, t]
        I.m = update_m_P_t_o2(I.H, I.J, I.m_p, I.C_p)
        I.D = update_D_P_t_o2(I.H, I.J, I.m, I.m_p, I.C_p)
        DJ =  (D_exp[:, :, t + 1] + np.einsum('i,j->ij',m_exp[:, t + 1],I.m_p,optimize=True)) - (I.D + np.einsum('i,j->ij',I.m,I.m_p,optimize=True))
        DH = m_exp[:, t + 1] - I.m
        DJ_mean += DJ / Nsample
        DH_mean += DH / Nsample

    I.J += etaJ * DJ_mean
    I.H += etaH * DH_mean
    print('P_t_o2', beta_ref, rep, np.mean(DH_mean**2),np.mean(DJ_mean**2), np.mean((H - I.H)**2), np.mean((J - I.J)**2))
    
    # Every few iterations, check errors in validation data.
    # If error does not decrease, early stop gradient descent
    if rep%validation_rep==0:
        error_H_prev = error_H
        error_J_prev = error_J
        DH_mean_v = np.zeros(size)
        DJ_mean_v = np.zeros((size, size))
        for c, t in enumerate(timesteps):
            I.m_p = m_exp_v[:, t]
            I.C_p = C_exp_v[:, :, t]
            I.m = update_m_P_t_o2(I.H, I.J, I.m_p, I.C_p)
            I.D = update_D_P_t_o2(I.H, I.J, I.m, I.m_p, I.C_p)
            DJ =  (D_exp_v[:, :, t+1] + np.einsum('i,j->ij',m_exp_v[:, t+1],I.m_p,optimize=True)) - (I.D + np.einsum('i,j->ij',I.m,I.m_p,optimize=True))
            DH = m_exp_v[:, t+1] - I.m
            DH_mean_v += DH / Nsample
            DJ_mean_v += DJ / Nsample
        error_H = np.mean(DH_mean_v**2)
        error_J = np.mean(DJ_mean_v**2)
        
        # Stop condition
        print('P_t_o2','Stop condition', np.mean(DH_mean**2),np.mean(DJ_mean**2),np.mean(DH_mean_v**2),np.mean(DJ_mean_v**2))
        if error_H > error_H_prev or error_J > error_J_prev or rep > max_rep:
            cond=False
        else:
            HP_t = I.H.copy()
            JP_t = I.J.copy()
    rep += 1
time_P_t = time.perf_counter() - time_start

#    print(np.mean(DH_mean**2),np.mean(DJ_mean**2))
#    plt.figure()
##    plt.plot([np.min(m_exp[:, t + 1]),np.max(m_exp[:, t + 1])],[np.min(m_exp[:, t + 1]),np.max(m_exp[:, t + 1])],'k')
#    plt.plot(m_exp[:, t + 1], I.m-m_exp[:, t + 1],'.')
#    plt.figure()
##    plt.plot([np.min(D_exp[:, :, t + 1]),np.max(D_exp[:, :, t + 1])],[np.min(D_exp[:, :, t + 1]),np.max(D_exp[:, :, t + 1])],'k')
#    plt.plot(D_exp[:, :, t + 1].flatten(),I.D.flatten()-D_exp[:, :, t + 1].flatten(),'.')


# Run Plefka2[t], order 2
time_start = time.perf_counter()
I = mf_ising(size)
error_J = 1
error_H = 1
for c, t in enumerate(timesteps):
    I.H += np.arctanh(m_exp[:, t]) / Nsample
#    I.H = H.copy()
#    I.J = J.copy()

rep = 0
cond=True
while cond:
    DH_mean = np.zeros(size)
    DJ_mean = np.zeros((size, size))
    for c, t in enumerate(timesteps):
        I.m_p = m_exp[:, t]
        I.C_p = C_exp[:, :, t]
        I.D_p = D_exp[:, :, t]
        I.m, I.D = update_D_P2_t_o2(I.H, I.J, I.m_p, I.C_p, I.D_p)
#            I.m = update_m_P_t_o2(I.H, I.J, I.m_p, I.C_p)
#            I.D = update_D_P_t_o2(I.H, I.J, I.m, I.m_p, I.C_p)
        DJ =  (D_exp[:, :, t + 1] + np.einsum('i,j->ij',m_exp[:, t + 1],I.m_p,optimize=True)) - (I.D + np.einsum('i,j->ij',I.m,I.m_p,optimize=True))
        DH = m_exp[:, t + 1] - I.m
        DJ_mean += DJ / Nsample
        DH_mean += DH / Nsample

    I.J += etaJ * DJ_mean
    I.H += etaH * DH_mean
    print('P2_t_o2', beta_ref, rep, np.mean(DH_mean**2),np.mean(DJ_mean**2), np.mean((H - I.H)**2), np.mean((J - I.J)**2))

    # Every few iterations, check errors in validation data.
    # If error does not decrease, early stop gradient descent
    if rep%validation_rep==0:
        error_H_prev = error_H
        error_J_prev = error_J
        DH_mean_v = np.zeros(size)
        DJ_mean_v = np.zeros((size, size))
        for c, t in enumerate(timesteps):
            I.m_p = m_exp_v[:, t]
            I.C_p = C_exp_v[:, :, t]
            I.D_p = D_exp_v[:, :, t]
            I.m, I.D = update_D_P2_t_o2(I.H, I.J, I.m_p, I.C_p, I.D_p)
            DJ =  (D_exp_v[:, :, t+1] + np.einsum('i,j->ij',m_exp_v[:, t+1],I.m_p,optimize=True)) - (I.D + np.einsum('i,j->ij',I.m,I.m_p,optimize=True))
            DH = m_exp_v[:, t+1] - I.m
            DH_mean_v += DH / Nsample
            DJ_mean_v += DJ / Nsample
        error_H = np.mean(DH_mean_v**2)
        error_J = np.mean(DJ_mean_v**2)
        
        # Stop condition
        print('P2_t_o2','Stop condition', np.mean(DH_mean**2),np.mean(DJ_mean**2),np.mean(DH_mean_v**2),np.mean(DJ_mean_v**2))
        if error_H > error_H_prev or error_J > error_J_prev or rep > max_rep:
            cond=False
        else:
            HP2_t = I.H.copy()
            JP2_t = I.J.copy()
    rep += 1
time_P2_t = time.perf_counter() - time_start


#    print(np.mean(DH_mean**2),np.mean(DJ_mean**2))
#    plt.figure()
##    plt.plot([np.min(m_exp[:, t + 1]),np.max(m_exp[:, t + 1])],[np.min(m_exp[:, t + 1]),np.max(m_exp[:, t + 1])],'k')
#    plt.plot(m_exp[:, t + 1], I.m-m_exp[:, t + 1],'.')
#    plt.figure()
##    plt.plot([np.min(D_exp[:, :, t + 1]),np.max(D_exp[:, :, t + 1])],[np.min(D_exp[:, :, t + 1]),np.max(D_exp[:, :, t + 1])],'k')
#    plt.plot(D_exp[:, :, t + 1].flatten(),I.D.flatten()-D_exp[:, :, t + 1].flatten(),'.')
#    plt.show()
#        
    
    
# Run Plefka[t-1], order 1
time_start = time.perf_counter()
I = mf_ising(size)
error_J = 1
error_H = 1
for c, t in enumerate(timesteps):
    I.H += np.arctanh(m_exp[:, t]) / Nsample

rep = 0
cond=True
while cond:
    DH = np.zeros(size)
    DJ = np.zeros((size, size))
    DH_mean = np.zeros(size)
    DJ_mean = np.zeros((size, size))
    for c, t in enumerate(timesteps):
        I.m_p = m_exp[:, t]
        I.C_p = C_exp[:, :, t]
        I.m = update_m_P_t1_o1(I.H, I.J, I.m_p)
        I.D = update_D_P_t1_o1(I.H, I.J, I.m_p, I.C_p)
        DJ =  (D_exp[:, :, t + 1] + np.einsum('i,j->ij',m_exp[:, t + 1],I.m_p,optimize=True)) - (I.D + np.einsum('i,j->ij',I.m,I.m_p,optimize=True))
        DH = m_exp[:, t + 1] - I.m
        DJ_mean += DJ / Nsample
        DH_mean += DH / Nsample

    I.J += etaJ * DJ_mean
    I.H += etaH * DH_mean

    print('P_t1_o1', beta_ref, rep, np.mean(DH_mean**2),np.mean(DJ_mean**2), np.mean((H - I.H)**2), np.mean((J - I.J)**2))
    
    # Every few iterations, check errors in validation data.
    # If error does not decrease, early stop gradient descent
    if rep%validation_rep==0:
        error_H_prev = error_H
        error_J_prev = error_J
        DH_mean_v = np.zeros(size)
        DJ_mean_v = np.zeros((size, size))
        for c, t in enumerate(timesteps):
            I.m_p = m_exp_v[:, t]
            I.C_p = C_exp_v[:, :, t]
            I.m = update_m_P_t1_o1(I.H, I.J, I.m_p)
            I.D = update_D_P_t1_o1(I.H, I.J, I.m_p, I.C_p)
            DJ =  (D_exp_v[:, :, t+1] + np.einsum('i,j->ij',m_exp_v[:, t+1],I.m_p,optimize=True)) - (I.D + np.einsum('i,j->ij',I.m,I.m_p,optimize=True))
            DH = m_exp_v[:, t+1] - I.m
            DH_mean_v += DH / Nsample
            DJ_mean_v += DJ / Nsample
        error_H = np.mean(DH_mean_v**2)
        error_J = np.mean(DJ_mean_v**2)
        
        # Stop condition
        print('P_t1_o1','Stop condition', np.mean(DH_mean**2),np.mean(DJ_mean**2),np.mean(DH_mean_v**2),np.mean(DJ_mean_v**2))
        if error_H > error_H_prev or error_J > error_J_prev or rep > max_rep:
            cond=False
        else:
            HP_t1 = I.H.copy()
            JP_t1 = I.J.copy()
    rep += 1
time_P_t1 = time.perf_counter() - time_start

filename = 'data/results/inverse_' + str(int(beta_ref * 100)) +'_R_' + str(R) +'_Nsample_'+str(Nsample)+'.npz'
np.savez_compressed(filename,
                    H=H, J=J,
                    HP_t1_t=HP_t1_t, HP_t=HP_t, HP_t1=HP_t1, HP2_t=HP2_t,
                    JP_t1_t=JP_t1_t, JP_t=JP_t, JP_t1=JP_t1, JP2_t=JP2_t,
                    time_P_t1_t=time_P_t1_t,
                    time_P_t=time_P_t,
                    time_P_t1=time_P_t1,
                    time_P2_t=time_P2_t)
