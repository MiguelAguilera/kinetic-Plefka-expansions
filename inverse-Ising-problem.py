#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:45:08 2018

@author: maguilera
"""

from mf_ising import mf_ising
from kinetic_ising import ising
from ising_functions import update_m_P2_o1, update_D_P2_o1, update_D_P2_o1, update_m_P1_o2, update_P1D_o2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

plt.rc('text', usetex=True)
font = {'size': 15}
plt.rc('font', **font)
plt.rc('legend', **{'fontsize': 12})


size=512
R = 500000
mode = 'c'
gamma1 = 0.5
gamma2 = 0.1

T = 128-1


cmap = cm.get_cmap('inferno_r')
colors=[]
for i in range(4):
    colors+=[cmap((i+0.5)/4)]

error_ref=0.0001
max_rep=5000

if gamma1==0.5 and gamma2==0.1:
    beta0 = 1.1123
B=21
betas = 1 + np.linspace(-1,1,B)*0.3
print(betas)

for ib in range(len(betas)):

    beta_ref = round(betas[ib],3)
    beta = beta_ref * beta0
    T=128
    #T=2

    filename = 'data/m-c-ts0-gamma1-' + str(gamma1) +'-gamma2-' + str(gamma2) + '-s-' + \
                str(size) + '-R-' + str(R) + '-beta-' + str(beta_ref) + '.npz'
    data = np.load(filename)
    H = data['H']
    J = data['J']

    iu1 = np.triu_indices(size, 1)
    s0 = np.array(data['s0'])


    eta=1
    error_ref=1E-5
    max_rep=1000
    max_rep_min=5
    #error_ref=0.05

    etaH=eta
    #etaJ=eta/size*5
    etaJ=eta/size**0.5
    st=16
    timesteps=np.arange(st//2,T//2-1,st)
    #timesteps=np.array([128-2])
    Nsample=len(timesteps)

    #print(Nsample)
    #exit()

    I = mf_ising(size)
    error=1
    min_error=error

    for c,t in enumerate(timesteps):
        I.H+=np.arctanh(data['m'][:, t])/Nsample
    rep=0
    rep_min=0

    while error>error_ref:
        DH=np.zeros(size)
        DJ=np.zeros((size,size))
        DH_mean=np.zeros(size)
        DJ_mean=np.zeros((size,size))
        for c,t in enumerate(timesteps):
            I.m = data['m'][:, t]
            I.C = data['C'][:, :, t]
            I.update_P0_o2()
            DJ=data['D'][:, :, t+1]-I.D
            DH=data['m'][:, t+1]-I.m
            DJ_mean+=DJ/Nsample
            DH_mean+=DH/Nsample
            
        error=max(np.max(np.abs(DH_mean)),np.max(np.abs(DJ_mean)))
        if error<min_error:
            HP0o2 = I.H.copy()
            JP0o2 = I.J.copy()
            min_error=error
            rep_min=0
        I.J+=etaJ*DJ_mean
        I.H+=etaH*DH_mean
        print('P0_o2',beta,rep,np.max(np.abs(DH_mean)),np.max(np.abs(DJ_mean)),np.max(np.abs(H-I.H)),np.max(np.abs(J-I.J)))
        rep+=1
        rep_min+=1
        if error<error_ref or rep>max_rep or (rep>(max_rep/10) and rep_min>max_rep_min):
            print(error<error_ref, rep>max_rep, rep_min>max_rep_min)
            break




    I = mf_ising(size)
    error=1
    min_error=error
    for c,t in enumerate(timesteps):
        I.H+=np.arctanh(data['m'][:, t])/Nsample
    rep=0
    rep_min=0
    while error>error_ref:
        DH=np.zeros(size)
        DJ=np.zeros((size,size))
        DH_mean=np.zeros(size)
        DJ_mean=np.zeros((size,size))
        for c,t in enumerate(timesteps):
            I.m = data['m'][:, t]
            I.C = data['C'][:, :, t]
            I.update_P1_o2()
            DJ=data['D'][:, :, t+1]-I.D
            DH=data['m'][:, t+1]-I.m
            DJ_mean+=DJ/Nsample
            DH_mean+=DH/Nsample
            
        error=max(np.max(np.abs(DH_mean)),np.max(np.abs(DJ_mean)))
        if error<min_error:
            HP1o2 = I.H.copy()
            JP1o2 = I.J.copy()
            min_error=error
            rep_min=0
        I.J+=etaJ*DJ_mean
        I.H+=etaH*DH_mean
        print('P1_o2',beta,rep,np.max(np.abs(DH_mean)),np.max(np.abs(DJ_mean)),np.max(np.abs(H-I.H)),np.max(np.abs(J-I.J)))
        rep+=1
        rep_min+=1
        if error<error_ref or rep>max_rep or (rep>(max_rep/10) and rep_min>max_rep_min):
            print(error<error_ref, rep>max_rep, rep_min>max_rep_min)
            break


    I = mf_ising(size)
    error=1
    min_error=error
    for c,t in enumerate(timesteps):
        I.H+=np.arctanh(data['m'][:, t])/Nsample
    rep=0
    rep_min=0
    while error>error_ref:
        DH_mean=np.zeros(size)
        DJ_mean=np.zeros((size,size))
        for c,t in enumerate(timesteps):
            I.m_p = data['m'][:, t]
            I.C_p = data['C'][:, :, t]
            I.D = data['D'][:, :, t]
            I.m,_,I.D = update_P1D_o2(I.H, I.J, data['m'][:, t], data['C'][:, :, t], data['D'][:, :, t])
            DJ=data['D'][:, :, t+1]-I.D
            DH=data['m'][:, t+1]-I.m
            DJ_mean+=DJ/Nsample
            DH_mean+=DH/Nsample
            
        error=max(np.max(np.abs(DH_mean)),np.max(np.abs(DJ_mean)))
        if error<min_error:
            HP1Co2 = I.H.copy()
            JP1Co2 = I.J.copy()
            min_error=error
            rep_min=0
        I.J+=etaJ*DJ_mean
        I.H+=etaH*DH_mean
        print('P1C_o2',beta,rep,np.max(np.abs(DH_mean)),np.max(np.abs(DJ_mean)),np.max(np.abs(H-I.H)),np.max(np.abs(J-I.J)))
        rep+=1
        rep_min+=1
        if error<error_ref or rep>max_rep or (rep>(max_rep/10) and rep_min>max_rep_min):
            print(error<error_ref, rep>max_rep, rep_min>max_rep_min)
            break



    I = mf_ising(size)
    error=1
    min_error=error
    for c,t in enumerate(timesteps):
        I.H+=np.arctanh(data['m'][:, t])/Nsample

    from ising_functions import integrate_1DGaussian, dT1_1


    rep=0
    rep_min=0
    while error>error_ref:
        DH=np.zeros(size)
        DJ=np.zeros((size,size))
        DH_mean=np.zeros(size)
        DJ_mean=np.zeros((size,size))
        for c,t in enumerate(timesteps):
            I.m = update_m_P2_o1(I.H, I.J, data['m'][:, t])
            I.D = update_D_P2_o1(I.H, I.J, data['m'][:, t], data['C'][:, :, t])
            DJ=data['D'][:, :, t+1]-I.D
            DH=data['m'][:, t+1]-I.m
            DJ_mean+=DJ/Nsample
            DH_mean+=DH/Nsample
            
        error=max(np.max(np.abs(DH_mean)),np.max(np.abs(DJ_mean)))
        if error<min_error:
            HP2o1 = I.H.copy()
            JP2o1 = I.J.copy()
            min_error=error
            rep_min=0
        I.J+=etaJ*DJ_mean
        I.H+=etaH*DH_mean
        
        print('P2_o1',beta,rep,np.max(np.abs(DH_mean)),np.max(np.abs(DJ_mean)),np.max(np.abs(H-I.H)),np.max(np.abs(J-I.J)))
        rep+=1
        rep_min+=1
        if error<error_ref or rep>max_rep or (rep>(max_rep/10) and rep_min>max_rep_min):
            print(error<error_ref, rep>max_rep, rep_min>max_rep_min)
            break
            

    filename='img/compare-J_' + str(int(beta_ref * 100)) +'.npz'
    np.savez_compressed(filename,
          H=H, J=J, 
          HP0o2=HP0o2, HP1o2=HP1o2, HP2o1= HP2o1, HP1Co2=HP1Co2,
          JP0o2=JP0o2, JP1o2=JP1o2, JP2o1= JP2o1, JP1Co2=JP1Co2)


