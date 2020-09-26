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
import time
from sys import argv

def nsf(num, n=4):
    """n-Significant Figures"""
    numstr = ("{0:.%ie}" % (n-1)).format(num)
    return float(numstr)

gamma1 = 0.5
gamma2 = 0.1


size = 512
R = 1000000

T = 128

max_rep = 10000

B=21
betas = 1 + np.linspace(-1, 1, B) * 0.3
betas=betas[betas>=1]
#betas=np.array([1.0])
B=len(betas)
for ib in range(B):

    beta_ref = round(betas[ib], 3)        # beta / beta_c value


    timesteps0 = np.arange(T-4,T-2)
    T0=len(timesteps0)
    shuffled_steps = np.random.permutation(T0)

    filename = 'data/data-gamma1-' + str(gamma1) + '-gamma2-' + str(
        gamma2) + '-s-' + str(size) + '-R-' + str(R) + '-beta-' + str(beta_ref) + '.npz'
    data = np.load(filename)
    H = data['H']
    J = data['J']
    m_exp=data['m']
    C_exp=data['C']
    D_exp=data['D']
    mp_exp = np.zeros((size,T))
    mp_exp[:,1:] = m_exp[:,0:-1] 
    mp_exp[:,0] = data['s0']
    c_exp = C_exp + np.einsum('it,jt->ijt',m_exp,m_exp,optimize=True)
    d_exp = D_exp + np.einsum('it,jt->ijt',m_exp,mp_exp,optimize=True)
    
    W=T-3
    dp_mean = np.mean(d_exp[:, :,T-W-1:T-1],axis=2)
    cp_mean = np.mean(c_exp[:, :,T-W-1:T-1],axis=2)
    mp_mean = np.mean(m_exp[:,T-W-1:T-1],axis=1)
    mpp_mean = np.mean(m_exp[:,T-W-2:T-2],axis=1)
    d_mean = np.mean(d_exp[:, :,T-W:T],axis=2)
    c_mean = np.mean(c_exp[:, :,T-W:T],axis=2)
    m_mean = np.mean(m_exp[:,T-W:T],axis=1)
    C_mean = c_mean - np.einsum('i,j->ij',m_mean,m_mean,optimize=True)
    D_mean = d_mean - np.einsum('i,j->ij',m_mean,mp_mean,optimize=True)
    Cp_mean = cp_mean - np.einsum('i,j->ij',mp_mean,mp_mean,optimize=True)
    Dp_mean = dp_mean - np.einsum('i,j->ij',mp_mean,mpp_mean,optimize=True)
#    
    s0 = np.array(data['s0'])
    del data

    # Reference error for finishing gradient descent
    ref_error = 1E-12
    print('ref error:',ref_error)

    # Learning speed parameters
    etaH = 1 / size**0.5
    etaJ = 1 / size**0.5
    

# Run Plefka[t-1,t], order 2

    time_start = time.perf_counter()
    I = mf_ising(size)
    I.H=np.zeros(size)
    I.J=np.zeros((size,size))
    rep = 0
    cond = True
    while cond:
        I.m_p = mp_mean
        I.m = update_m_P_t1_t_o2(I.H, I.J, I.m_p)
        I.D = update_D_P_t1_t_o2(I.H, I.J, I.m, I.m_p)
#                DJ =  (D_exp[:, :, t + 1] + np.einsum('i,j->ij',m_exp[:, t + 1],I.m_p,optimize=True)) - (I.D + np.einsum('i,j->ij',I.m,I.m_p,optimize=True))
        DJ =  D_mean - I.D 
        DH = m_mean - I.m
        I.J += etaJ * DJ
        I.H += etaH * DH
        I.H=np.clip(I.H,-5,5)
        error_H=np.mean(DH**2)
        error_J=np.mean(DJ**2)
        print('P_t1_t_o2', beta_ref, rep, nsf(error_H), nsf(error_J), nsf(np.mean((H - I.H)**2)), nsf(np.mean((J - I.J)**2)))
        rep += 1
        if rep >= max_rep or ((error_H<ref_error) and (error_J<ref_error)):
                cond=False
    HP_t1_t = I.H.copy()
    JP_t1_t = I.J.copy()
    time_P_t1_t = time.perf_counter() - time_start


# Run Plefka[t], order 2

    time_start = time.perf_counter()
    I = mf_ising(size)
    I.H=np.zeros(size)
    I.J=np.zeros((size,size))
    rep = 0
    cond = True
    while cond:
        I.m_p = mp_mean
        I.C_p = Cp_mean
        I.m = update_m_P_t_o2(I.H, I.J, I.m_p, I.C_p)
        I.D = update_D_P_t_o2(I.H, I.J, I.m, I.m_p, I.C_p)
#            DJ =  (D_exp[:, :, t + 1] + np.einsum('i,j->ij',m_exp[:, t + 1],m_exp[:, t],optimize=True)) - (I.D + np.einsum('i,j->ij',I.m,m_exp[:, t],optimize=True))
        DJ =  D_mean - I.D 
        DH = m_mean - I.m
        I.J += etaJ * DJ
        I.H += etaH * DH
        error_H=np.mean(DH**2)
        error_J=np.mean(DJ**2)
        print('P_t_o2', beta_ref,  rep, nsf(error_H), nsf(error_J), nsf(np.mean((H - I.H)**2)), nsf(np.mean((J - I.J)**2)))
        rep += 1
        if rep >= max_rep or ((error_H<ref_error) and (error_J<ref_error)):
                cond=False
    HP_t = I.H.copy()
    JP_t = I.J.copy()
    time_P_t = time.perf_counter() - time_start



    # Run Plefka2[t], order 2
    time_start = time.perf_counter()
    I = mf_ising(size)
    I.H=np.zeros(size)
    I.J=np.zeros((size,size))
    rep = 0
    cond = True
    while cond:
        I.m_p = mp_mean
        I.C_p = Cp_mean
        I.D_p = Dp_mean
        I.m, I.D = update_D_P2_t_o2(I.H, I.J, I.m_p, I.C_p, I.D_p)
        DJ =  D_mean - I.D 
        DH = m_mean - I.m
        I.J += etaJ * DJ
        I.H += etaH * DH
        error_H=np.mean(DH**2)
        error_J=np.mean(DJ**2)
        print('P2_t_o2', beta_ref,  rep, nsf(error_H), nsf(error_J), nsf(np.mean((H - I.H)**2)), nsf(np.mean((J - I.J)**2)))
        rep += 1
        if rep >= max_rep or ((error_H<ref_error) and (error_J<ref_error)):
                cond=False
    HP2_t = I.H.copy()
    JP2_t = I.J.copy()
    time_P2_t = time.perf_counter() - time_start

        
    # Run Plefka[t-1], order 1
    time_start = time.perf_counter()
    I = mf_ising(size)
    I.H=np.zeros(size)
    I.J=np.zeros((size,size))
    rep = 0
    cond = True
    while cond:
        I.m_p = mp_mean
        I.C_p = Cp_mean
        I.m = update_m_P_t1_o1(I.H, I.J, I.m_p)
        I.D = update_D_P_t1_o1(I.H, I.J, I.m_p, I.C_p)
#                DJ =  (D_exp[:, :, t + 1] + np.einsum('i,j->ij',m_exp[:, t + 1],I.m_p,optimize=True)) - (I.D + np.einsum('i,j->ij',I.m,I.m_p,optimize=True))
        DJ =  D_mean - I.D 
        DH = m_mean - I.m
        I.J += etaJ * DJ
        I.H += etaH * DH
        error_H=np.mean(DH**2)
        error_J=np.mean(DJ**2)
        print('P_t1_o1', beta_ref,  rep, nsf(error_H), nsf(error_J), nsf(np.mean((H - I.H)**2)), nsf(np.mean((J - I.J)**2)))
        rep += 1
        if rep >= max_rep or ((error_H<ref_error) and (error_J<ref_error)):
                cond=False
    HP_t1 = I.H.copy()
    JP_t1 = I.J.copy()
    time_P_t1 = time.perf_counter() - time_start
    

    filename = 'data/results/inverse_' + str(int(beta_ref * 100)) +'_R_' + str(R) + '.npz'
    print(filename)
    np.savez_compressed(filename,
                        H=H, J=J,
                        HP_t1_t=HP_t1_t, HP_t=HP_t, HP_t1=HP_t1, HP2_t=HP2_t,
                        JP_t1_t=JP_t1_t, JP_t=JP_t, JP_t1=JP_t1, JP2_t=JP2_t,
                        time_P_t1_t=time_P_t1_t,
                        time_P_t=time_P_t,
                        time_P_t1=time_P_t1,
                        time_P2_t=time_P2_t)
                        
