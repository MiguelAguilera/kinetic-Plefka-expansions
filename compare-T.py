#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:45:08 2018

@author: maguilera
"""

from mf_ising import mf_ising, mf_ising_roudi
from kinetic_ising import ising
import numpy as np
from matplotlib import pyplot as plt


plt.rc('text', usetex=True)
font = {'size':15}
plt.rc('font',**font)
plt.rc('legend',**{'fontsize':12})

size = 100
beta = 0.9
#beta = 1.2
beta = 1.0
R = 10000
mode = 'c'
random_s0=False
random_s0=True

T = 32
#T=8
#T=2
iu1 = np.triu_indices(size, 1)

mP0o2_mean = np.zeros(T)
mP1o2_mean = np.zeros(T)
mP1Co2_mean = np.zeros(T)
mPexp_mean = np.zeros(T)
mP0o2_std = np.zeros(T)
mP1o2_std = np.zeros(T)
mP1Co2_std = np.zeros(T)
mP2o1_mean = np.zeros(T)
mP2o2_mean = np.zeros(T)
mP2o1_std = np.zeros(T)
mP2o2_std = np.zeros(T)
mPexp_std = np.zeros(T)

CP0o2_mean = np.zeros(T)
CP1o2_mean = np.zeros(T)
CP1Co2_mean = np.zeros(T)
CPexp_mean = np.zeros(T)
CP0o2_std = np.zeros(T)
CP1o2_std = np.zeros(T)
CP1Co2_std = np.zeros(T)
CP2o1_mean = np.zeros(T)
CP2o2_mean = np.zeros(T)
CP2o1_std = np.zeros(T)
CP2o2_std = np.zeros(T)
CPexp_std = np.zeros(T)

DP0o2_mean = np.zeros(T)
DP1o2_mean = np.zeros(T)
DP1Co2_mean = np.zeros(T)
DPexp_mean = np.zeros(T)
DP0o2_std = np.zeros(T)
DP1o2_std = np.zeros(T)
DP1Co2_std = np.zeros(T)
DP2o1_mean = np.zeros(T)
DP2o2_mean = np.zeros(T)
DP2o1_std = np.zeros(T)
DP2o2_std = np.zeros(T)
DPexp_std = np.zeros(T)

if random_s0:
    filename = 'data/m-' + mode + '-rs0-s-' + str(size) + '-R-' + str(R) + '-beta-' + str(beta) + '.npz'
else:
    filename = 'data/m-' + mode + '-ss0-s-' + str(size) + '-R-' + str(R) + '-beta-' + str(beta) + '.npz'
data = np.load(filename)
H = data['H']
J = data['J']
s0 = data['s0']
if not s0 is np.ndarray:
    s0=None
#	m_exp = data['m'][:, T]
#	C_exp_T = data['C'][:, :, T]
for t in range(T):
	C_exp = data['C'][:, :, t+1]
	CPexp_mean[t] = np.mean(C_exp[iu1])
	CPexp_std[t] = np.std(C_exp[iu1])

print('TAP')
I = mf_ising(size)
I.H = H.copy()
I.J = J.copy()

I.initialize_state(s0)
for t in range(T):
	I.update_P0_o2()
	mP0o2 = I.m.copy()
	CP0o2 = I.C.copy()
	DP0o2 = I.D.copy()
	CP0o2_mean[t] = np.mean(CP0o2[iu1]-data['C'][:, :, t+1][iu1])
	CP0o2_std[t] = np.std(CP0o2[iu1]-data['C'][:, :, t+1][iu1])
	mP0o2_mean[t] = np.mean(mP0o2-data['m'][:, t+1])
	mP0o2_std[t] = np.std(mP0o2-data['m'][:, t+1])
	DP0o2_mean[t] = np.mean(DP0o2-data['D'][:,:, t+1])
	DP0o2_std[t] = np.std(DP0o2-data['D'][:,:, t+1])
	
I.initialize_state(s0)
for t in range(T):
	I.update_P1_o2()
	mP1o2 = I.m.copy()
	CP1o2 = I.C.copy()
	DP1o2 = I.D.copy()
	CP1o2_mean[t] = np.mean(CP1o2[iu1]-data['C'][:, :, t+1][iu1])
	CP1o2_std[t] = np.std(CP1o2[iu1]-data['C'][:, :, t+1][iu1])
	mP1o2_mean[t] = np.mean(mP1o2-data['m'][:, t+1])
	mP1o2_std[t] = np.std(mP1o2-data['m'][:, t+1])
	DP1o2_mean[t] = np.mean(DP1o2-data['D'][:,:, t+1])
	DP1o2_std[t] = np.std(DP1o2-data['D'][:,:, t+1])
	
	
I.initialize_state(s0)
for t in range(T):
	I.update_P1C_o2()
	mP1Co2 = I.m.copy()
	CP1Co2 = I.C.copy()
	DP1Co2 = I.D.copy()
	CP1Co2_mean[t] = np.mean(CP1Co2[iu1]-data['C'][:, :, t+1][iu1])
	CP1Co2_std[t] = np.std(CP1Co2[iu1]-data['C'][:, :, t+1][iu1])
	mP1Co2_mean[t] = np.mean(mP1Co2-data['m'][:, t+1])
	mP1Co2_std[t] = np.std(mP1Co2-data['m'][:, t+1])
	DP1Co2_mean[t] = np.mean(DP1Co2-data['D'][:,:, t+1])
	DP1Co2_std[t] = np.std(DP1Co2-data['D'][:,:, t+1])

	
I.initialize_state(s0)
for t in range(T):
	I.update_P2_o1()
	mP2o1 = I.m.copy()
	CP2o1 = I.C.copy()
	DP2o1 = I.D.copy()
	CP2o1_mean[t] = np.mean(CP2o1[iu1]-data['C'][:, :, t+1][iu1])
	CP2o1_std[t] = np.std(CP2o1[iu1]-data['C'][:, :, t+1][iu1])
	mP2o1_mean[t] = np.mean(mP2o1-data['m'][:, t+1])
	mP2o1_std[t] = np.std(mP2o1-data['m'][:, t+1])
	DP2o1_mean[t] = np.mean(DP2o1[iu1]-data['D'][:, :, t+1])
	DP2o1_std[t] = np.std(DP2o1[iu1]-data['D'][:, :, t+1])
#I.initialize_state(s0)
#for t in range(T):
#	I.update_P2_o2()
#	mP2o2 = I.m.copy()
#	CP2o2 = I.C.copy()
#	P2o2_mean[t] = np.mean(CP2o2[iu1]-data['C'][:, :, t+1][iu1])
#	P2o2_std[t] = np.std(CP2o2[iu1]-data['C'][:, :, t+1][iu1])


steps=np.arange(T)+1

fig, ax = plt.subplots(1,1,figsize=(5,4))
plt.plot(steps,mP0o2_mean,'b',label='P[t-1:t]')
plt.fill_between(steps,mP0o2_mean-mP0o2_std,mP0o2_mean+mP0o2_std, color='b',alpha=0.25)
plt.plot(steps,mP1o2_mean,'g',label='P[t]')
plt.fill_between(steps,mP1o2_mean-mP1o2_std,mP1o2_mean+mP1o2_std, color='g',alpha=0.25)
plt.plot(steps,mP1Co2_mean,'r',label='P2[C]')
plt.fill_between(steps,mP1Co2_mean-mP1Co2_std,mP1Co2_mean+mP1Co2_std, color='r',alpha=0.25)
plt.plot(steps,mP2o1_mean,'c',label='P[t-1]')
plt.fill_between(steps,mP2o1_mean-mP2o1_std,mP2o1_mean+mP2o1_std, color='c',alpha=0.25)
plt.plot(steps,steps*0,'k')
plt.title(r'$\beta='+str(beta)+r'$')
plt.xlabel(r'$t$')
plt.ylabel(r'$\epsilon_C$')
plt.legend()



fig, ax = plt.subplots(1,1,figsize=(5,4))
plt.plot(steps,CP0o2_mean,'b',label='P[t-1:t]')
plt.fill_between(steps,CP0o2_mean-CP0o2_std,CP0o2_mean+CP0o2_std, color='b',alpha=0.25)
plt.plot(steps,CP1o2_mean,'g',label='P[t]')
plt.fill_between(steps,CP1o2_mean-CP1o2_std,CP1o2_mean+CP1o2_std, color='g',alpha=0.25)
plt.plot(steps,CP1Co2_mean,'r',label='P2[C]')
plt.fill_between(steps,CP1Co2_mean-CP1Co2_std,CP1Co2_mean+CP1Co2_std, color='r',alpha=0.25)
plt.plot(steps,CP2o1_mean,'c',label='P[t-1]')
plt.fill_between(steps,CP2o1_mean-CP2o1_std,CP2o1_mean+CP2o1_std, color='c',alpha=0.25)
#plt.plot(steps,P2o2_mean,'m',label='P[t-1]')
#plt.fill_between(steps,P2o2_mean-P2o2_std,P2o2_mean+P2o2_std, color='m',alpha=0.25)
plt.plot(steps,steps*0,'k')
plt.title(r'$\beta='+str(beta)+r'$')
plt.xlabel(r'$t$')
plt.ylabel(r'$\epsilon_C$')
plt.legend()


fig, ax = plt.subplots(1,1,figsize=(5,4))
plt.plot(steps,DP0o2_mean,'b',label='P[t-1:t]')
plt.fill_between(steps,DP0o2_mean-DP0o2_std,DP0o2_mean+DP0o2_std, color='b',alpha=0.25)
plt.plot(steps,DP1o2_mean,'g',label='P[t]')
plt.fill_between(steps,DP1o2_mean-DP1o2_std,DP1o2_mean+DP1o2_std, color='g',alpha=0.25)
plt.plot(steps,DP1Co2_mean,'r',label='P2[D]')
plt.fill_between(steps,DP1Co2_mean-DP1Co2_std,DP1Co2_mean+DP1Co2_std, color='r',alpha=0.25)
plt.plot(steps,DP2o1_mean,'c',label='P[t-1]')
plt.fill_between(steps,DP2o1_mean-DP2o1_std,DP2o1_mean+DP2o1_std, color='c',alpha=0.25)
plt.plot(steps,steps*0,'k')
plt.title(r'$\beta='+str(beta)+r'$')
plt.xlabel(r'$t$')
plt.ylabel(r'$\epsilon_D$')
plt.legend()


plt.show()

