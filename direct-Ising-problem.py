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

# size = 200


size = 16
size= 32
size = 64
size = 128
size = 256
size=512

#gamma1 = 0.8
gamma1 = 0.5

#gamma2 = 0.2
gamma2 = 0.1

random_s0 = False
#stationary=True
stationary = False
# random_s0 = True

T = 64
#T=25
T = 128
#T=3

R = 100000

cmap = cm.get_cmap('inferno_r')
colors=[]
for i in range(4):
    colors+=[cmap((i+0.5)/4)]
    
mode = 'c'
if gamma1==0.5 and gamma2==0.1:
	beta0 = 1.1123
B=11


betas = 1 + np.linspace(-1,1,B)*0.3
#for beta in [0.7,0.8,0.9,1.0,1.1,1.2,1.3]:
for ib in range(len(betas)):
	beta_ref = round(betas[ib],3)
	beta = beta_ref * beta0

	# T = 16
	# T=2
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
	mP2o2_mean = np.ones(T + 1)
	mP0o2_std = np.zeros(T + 1)
	mP1o2_std = np.zeros(T + 1)
	mP1Co2_std = np.zeros(T + 1)
	mP2o1_std = np.zeros(T + 1)
	mP2o2_std = np.zeros(T + 1)
	mPexp_std = np.zeros(T + 1)

	CP0o2_mean = np.zeros(T + 1)
	CP1o2_mean = np.zeros(T + 1)
	CP1Co2_mean = np.zeros(T + 1)
	CPexp_mean = np.zeros(T + 1)
	CP0o2_std = np.zeros(T + 1)
	CP1o2_std = np.zeros(T + 1)
	CP1Co2_std = np.zeros(T + 1)
	CP2o1_mean = np.zeros(T + 1)
	CP2o2_mean = np.zeros(T + 1)
	CP2o1_std = np.zeros(T + 1)
	CP2o2_std = np.zeros(T + 1)
	CPexp_std = np.zeros(T + 1)

	DP0o2_mean = np.zeros(T + 1)
	DP1o2_mean = np.zeros(T + 1)
	DP1Co2_mean = np.zeros(T + 1)
	DPexp_mean = np.zeros(T + 1)
	DP0o2_std = np.zeros(T + 1)
	DP1o2_std = np.zeros(T + 1)
	DP1Co2_std = np.zeros(T + 1)
	DP2o1_mean = np.zeros(T + 1)
	DP2o2_mean = np.zeros(T + 1)
	DP2o1_std = np.zeros(T + 1)
	DP2o2_std = np.zeros(T + 1)
	DPexp_std = np.zeros(T + 1)

	folder = 'data'
	# folder='data (plefka)'
	# for folder in ['data','data (plefka)']:
	if True:
		if random_s0:
		    tmode = 'rs0'
		else:
		    tmode = 'ss0'
		    if not stationary:
		        tmode = 'ts0'
		filename = 'data/m-' + mode + '-'+tmode+'-gamma1-' + str(gamma1) +'-gamma2-' + str(gamma2) + '-s-' + \
		        str(size) + '-R-' + str(R) + '-beta-' + str(beta_ref) + '.npz'
	#    filename = 'data (plefka)/m-' + mode + '-' + tmode + '-s-' + \
	#            str(size) + '-R-' + str(R) + '-beta-' + str(beta) + '.npz'
	#    if random_s0:
	#        filename = folder + '/m-' + mode + '-rs0-s-' + \
	#            str(size) + '-R-' + str(R) + '-beta-' + str(beta) + '.npz'
	#    else:
	#        filename = folder + '/m-' + mode + '-ss0-s-' + \
	#            str(size) + '-R-' + str(R) + '-beta-' + str(beta) + '.npz'
	#        if not stationary:
	#            filename = folder + '/m-' + mode + '-ts0-s-' + \
	#                str(size) + '-R-' + str(R) + '-beta-' + str(beta) + '.npz'
		print(filename)
		data = np.load(filename)
		H = data['H']
		J = data['J']
		s0 = data['s0']
#		print(s0)
		# exit()

#		H1 = 0.3 * np.random.randn(size) * beta
#		plt.figure()
#		plt.subplot(311)
#		plt.hist(H, 10)
#		plt.subplot(312)
#		plt.hist(H1, 10)
#		plt.subplot(313)
#		plt.hist(J.flatten(), 50)
		

		if random_s0:
		    s0 = None
		else:
		    print(np.mean(s0))
		    #	m_exp = data['m'][:, T]
		    #	C_exp_T = data['C'][:, :, T]
		mPexp_mean[0] = 1
		for t in range(T):
		    print(beta_ref,beta,'Exp', str(t) + '/' + str(T))
		    m_exp = data['m'][:, t]
		    mPexp_mean[t + 1] = np.mean(m_exp)
		    mPexp_std[t + 1] = np.std(m_exp)
		    C_exp = data['C'][:, :, t]
		    CPexp_mean[t + 1] = np.mean(C_exp[iu1])
		    CPexp_std[t + 1] = np.std(C_exp[iu1])
		    D_exp = data['D'][:, :, t]
		    DPexp_mean[t + 1] = np.mean(D_exp)
		    DPexp_std[t + 1] = np.std(D_exp)
		    print(DPexp_mean[t + 1])

		plt.figure()
		plt.subplot(411)
		plt.plot(mPexp_mean)
		plt.subplot(414)
		plt.loglog(mPexp_mean-np.min(mPexp_mean))
	#
	#	plt.figure()
	#	plt.subplot(311)
	#	plt.plot(mPexp_mean)
		plt.subplot(412)
		plt.plot(CPexp_mean)
		plt.subplot(413)
		plt.plot(DPexp_mean)


	print(np.mean(H))
	steps = np.arange(T + 1)
	plt.figure()
	plt.plot(steps, mPexp_mean, 'k', label=r'$P$')
	plt.figure()
	plt.plot(steps, DPexp_mean, 'k', label=r'$P$')
	#plt.show()
	#exit()
	#    plt.show()
	# plt.show()
	# exit()
	print('TAP')
	I = mf_ising(size)
	I.H = H.copy()
	I.J = J.copy()

	I.initialize_state(s0)
	for t in range(T):
		print(beta,'P0_o2', str(t) + '/' + str(T))
		I.update_P0_o2()
		CP0o2_mean[t + 1] = np.mean(I.C[iu1])  # - data['C'][:, :, t][iu1])
		CP0o2_std[t + 1] = np.std(I.C[iu1])  # - data['C'][:, :, t][iu1])
		mP0o2_mean[t + 1] = np.mean(I.m)  # - data['m'][:, t])
		mP0o2_std[t + 1] = np.std(I.m)  # - data['m'][:, t])
		DP0o2_mean[t + 1] = np.mean(I.D)  # - data['D'][:, :, t])
		DP0o2_std[t + 1] = np.std(I.D)  # - data['D'][:, :, t])
		EmP0o2[t + 1] = np.mean((I.m - data['m'][:, t])**2)
		ECP0o2[t + 1] = np.mean((I.C[iu1] - data['C'][:, :, t][iu1])**2)
		EDP0o2[t + 1] = np.mean((I.D - data['D'][:, :, t])**2)
	I.initialize_state(s0)
	for t in range(T):
		print(beta,'P1_o2', str(t) + '/' + str(T))
		I.update_P1_o2()
	#    mP1o2 = I.m.copy()
	#    CP1o2 = I.C.copy()
	#    DP1o2 = I.D.copy()
		CP1o2_mean[t + 1] = np.mean(I.C[iu1])  # - data['C'][:, :, t][iu1])
		CP1o2_std[t + 1] = np.std(I.C[iu1])  # - data['C'][:, :, t][iu1])
		mP1o2_mean[t + 1] = np.mean(I.m)  # - data['m'][:, t])
		mP1o2_std[t + 1] = np.std(I.m)  # - data['m'][:, t])
		DP1o2_mean[t + 1] = np.mean(I.D)  # - data['D'][:, :, t])
		DP1o2_std[t + 1] = np.std(I.D)  # - data['D'][:, :, t])
		EmP1o2[t + 1] = np.mean((I.m - data['m'][:, t])**2)
		ECP1o2[t + 1] = np.mean((I.C[iu1] - data['C'][:, :, t][iu1])**2)
		EDP1o2[t + 1] = np.mean((I.D - data['D'][:, :, t])**2)
		print(mP1o2_mean[t + 1], CP1o2_mean[t + 1], DP1o2_mean[t + 1])

	I.initialize_state(s0)
	for t in range(T):
		print(beta,'P1C_o2', str(t) + '/' + str(T))
		I.update_P1C_o2()
		CP1Co2_mean[t + 1] = np.mean(I.C[iu1])  # - data['C'][:, :, t][iu1])
		CP1Co2_std[t + 1] = np.std(I.C[iu1])  # - data['C'][:, :, t][iu1])
		mP1Co2_mean[t + 1] = np.mean(I.m)  # - data['m'][:, t])
		mP1Co2_std[t + 1] = np.std(I.m)  # - data['m'][:, t])
		DP1Co2_mean[t + 1] = np.mean(I.D)  # - data['D'][:, :, t])
		DP1Co2_std[t + 1] = np.std(I.D)  # - data['D'][:, :, t])
		EmP1Co2[t + 1] = np.mean((I.m - data['m'][:, t])**2)
		ECP1Co2[t + 1] = np.mean((I.C[iu1] - data['C'][:, :, t][iu1])**2)
		EDP1Co2[t + 1] = np.mean((I.D - data['D'][:, :, t])**2)

	I.initialize_state(s0)
	for t in range(T):
		print(beta,'P2_o1', str(t) + '/' + str(T))
		I.update_P2_o1()
		CP2o1_mean[t + 1] = np.mean(I.C[iu1])  # - data['C'][:, :, t][iu1])
		CP2o1_std[t + 1] = np.std(I.C[iu1])  # - data['C'][:, :, t][iu1])
		mP2o1_mean[t + 1] = np.mean(I.m)  # - data['m'][:, t])
		mP2o1_std[t + 1] = np.std(I.m)  # - data['m'][:, t])
		DP2o1_mean[t + 1] = np.mean(I.D)  # - data['D'][:, :, t])
		DP2o1_std[t + 1] = np.std(I.D)  # - data['D'][:, :, t])
		EmP2o1[t + 1] = np.mean((I.m - data['m'][:, t])**2)
		ECP2o1[t + 1] = np.mean((I.C[iu1] - data['C'][:, :, t][iu1])**2)
		EDP2o1[t + 1] = np.mean((I.D - data['D'][:, :, t])**2)
	# I.initialize_state(s0)
	# for t in range(T):
	#	I.update_P2_o2()
	#	mP2o2 = I.m.copy()
	#	CP2o2 = I.C.copy()
	#	P2o2_mean[t] = np.mean(CP2o2[iu1]-data['C'][:, :, t+1][iu1])
	#	P2o2_std[t] = np.std(CP2o2[iu1]-data['C'][:, :, t+1][iu1])

	filename='img/compare-T_' + str(int(beta_ref * 100)) + '_size_'+str(size)+'.npz'
	np.savez_compressed(filename,
		  m_exp=data['m'][:, t], C_exp=data['C'][:, :, t],D_exp=data['D'][:, :, t],
		  mP0o2_mean=mP0o2_mean,mP1o2_mean=mP1o2_mean, mP2o1_mean=mP2o1_mean,mP1Co2_mean=mP1Co2_mean,
		  CP0o2_mean=CP0o2_mean,CP1o2_mean=CP1o2_mean, CP2o1_mean=CP2o1_mean,CP1Co2_mean=CP1Co2_mean,
		  DP0o2_mean=DP0o2_mean,DP1o2_mean=DP1o2_mean, DP2o1_mean=DP2o1_mean,DP1Co2_mean=DP1Co2_mean,
		  mP0o2_std=mP0o2_std,mP1o2_std=mP1o2_std, mP2o1_std=mP2o1_std,mP1Co2_std=mP1Co2_std,
		  CP0o2_std=CP0o2_std,CP1o2_std=CP1o2_std, CP2o1_std=CP2o1_std,CP1Co2_std=CP1Co2_std,
		  DP0o2_std=DP0o2_std,DP1o2_std=DP1o2_std, DP2o1_std=DP2o1_std,DP1Co2_std=DP1Co2_std,
	#      mPexp_mean=mPexp_mean,CPexp_mean=CPexp_mean,DPexp_mean=DPexp_mean,
	#      mP0o2=mP0o2,mP1o2=mP1o2,mP2o1=mP2o1,mP1Co2=mP1Co2,
	#      CP0o2=CP0o2,CP1o2=CP1o2,CP2o1=CP2o1,CP1Co2=CP1Co2,
	#      DP0o2=DP0o2,DP1o2=DP1o2,DP2o1=DP2o1,DP1Co2=DP1Co2,
		  EmP0o2=EmP0o2,EmP1o2=EmP1o2,EmP2o1=EmP2o1,EmP1Co2=EmP1Co2,
		  ECP0o2=ECP0o2,ECP1o2=ECP1o2,ECP2o1=ECP2o1,ECP1Co2=ECP1Co2,
		  EDP0o2=EDP0o2,EDP1o2=EDP1o2,EDP2o1=EDP2o1,EDP1Co2=EDP1Co2)

	steps = np.arange(T + 1)
	print('betas',beta_ref,beta0,beta,beta_ref*beta0)
	fig, ax = plt.subplots(1, 1, figsize=(5, 4))
	plt.plot(steps, EmP0o2, 'b', label=r'P[t-1:t]')
	plt.plot(steps, EmP1o2, 'g', label=r'P[t]')
	plt.plot(steps, EmP2o1, 'm', label=r'P[t-1]')
	plt.plot(steps, EmP1Co2, 'r', label=r'P[C]')
	plt.title(r'$\beta=' + str(beta) + r'$', fontsize=18)
	plt.xlabel(r'$t$', fontsize=18)
	plt.ylabel(r'$MSE[\textbf{m}_t]$', fontsize=18, rotation=0, labelpad=30)
	plt.legend()
#	plt.savefig('img/error_m-beta_' + str(int(beta_ref * 100)) +
#		        '.pdf', bbox_inches='tight')


	fig, ax = plt.subplots(1, 1, figsize=(5, 4))
	plt.plot(steps, ECP0o2, 'b', label=r'P[t-1:t]')
	plt.plot(steps, ECP1o2, 'g', label=r'P[t]')
	plt.plot(steps, ECP2o1, 'm', label=r'P[t-1]')
	plt.plot(steps, ECP1Co2, 'r', label=r'P[C]')
	plt.title(r'$\beta=' + str(beta) + r'$', fontsize=18)
	plt.xlabel(r'$t$', fontsize=18)
	plt.ylabel(r'$MSE[\textbf{C}_t]$', fontsize=18, rotation=0, labelpad=30)
	plt.legend()
#	plt.savefig('img/error_C-beta_' + str(int(beta_ref * 100)) +
#		        '.pdf', bbox_inches='tight')

	fig, ax = plt.subplots(1, 1, figsize=(5, 4))
	plt.plot(steps, EDP0o2, 'b', label=r'P[t-1:t]')
	plt.plot(steps, EDP1o2, 'g', label=r'P[t]')
	plt.plot(steps, EDP2o1, 'm', label=r'P[t-1]')
	plt.plot(steps, EDP1Co2, 'r', label=r'P[D]')
	plt.title(r'$\beta=' + str(beta) + r'$', fontsize=18)
	plt.xlabel(r'$t$', fontsize=18)
	plt.ylabel(r'$MSE[\textbf{D}_t]$', fontsize=18, rotation=0, labelpad=30)
	plt.legend()
#	plt.savefig('img/error_D-beta_' + str(int(beta_ref * 10)) +
#		        '.pdf', bbox_inches='tight')

	fig, ax = plt.subplots(1, 1, figsize=(5, 4))
	plt.plot(steps, mP0o2_mean, 'v', color=colors[0], ms=3, label=r'P[t-1:t]')
	plt.plot(steps, mP1o2_mean, 's', color=colors[1], ms=3, label=r'P[t]')
	plt.plot(steps, mP2o1_mean, 'd', color=colors[2], ms=3, label=r'P[t-1]')
	plt.plot(steps, mP1Co2_mean, 'o', color=colors[3], ms=3, label=r'P[C]')
	plt.plot(steps, mPexp_mean, 'k', label=r'$P$')
	plt.title(r'$\beta/\beta_c=' + str(beta_ref) + r'$', fontsize=18)
	plt.xlabel(r'$t$', fontsize=18)
	plt.ylabel(r'$\langle \textbf{m}_t \rangle$', fontsize=18, rotation=0, labelpad=15)
	plt.legend()
	plt.savefig('img/evolution_m-beta_' + str(int(beta_ref * 100)) +
		        '.pdf', bbox_inches='tight')


	fig, ax = plt.subplots(1, 1, figsize=(5, 4))
	plt.plot(steps, CP0o2_mean, 'v', color=colors[0], ms=3, label=r'P[t-1:t]')
	plt.plot(steps, CP1o2_mean, 's', color=colors[1], ms=3, label=r'P[t]')
	plt.plot(steps, CP2o1_mean, 'd', color=colors[2], ms=3, label=r'P[t-1]')
	plt.plot(steps, CP1Co2_mean, 'o', color=colors[3], ms=3, label=r'P[C]')
	plt.plot(steps, CPexp_mean, 'k', label=r'$P$')
	plt.title(r'$\beta/\beta_c=' + str(beta_ref) + r'$', fontsize=18)
	plt.xlabel(r'$t$', fontsize=18)
	plt.ylabel(r'$\langle \textbf{C}_t \rangle$', fontsize=18, rotation=0, labelpad=15)
	plt.legend()
	#plt.axis([0,T,0,1])
	plt.savefig('img/evolution_C-beta_' + str(int(beta_ref * 100)) +
		        '.pdf', bbox_inches='tight')

	fig, ax = plt.subplots(1, 1, figsize=(5, 4))
	plt.plot(steps, DP0o2_mean, 'v', color=colors[0], ms=3, label=r'P[t-1:t]')
	plt.plot(steps, DP1o2_mean, 's', color=colors[1], ms=3, label=r'P[t]')
	plt.plot(steps, DP2o1_mean, 'd', color=colors[2], ms=3, label=r'P[t-1]')
	plt.plot(steps, DP1Co2_mean, 'o', color=colors[3], ms=3, label=r'P[D]')
	plt.plot(steps, DPexp_mean, 'k', label=r'$P$')
	plt.title(r'$\beta/\beta_c=' + str(beta_ref) + r'$', fontsize=18)
	plt.xlabel(r'$t$', fontsize=18)
	plt.ylabel(r'$\langle \textbf{D}_t \rangle$', fontsize=18, rotation=0, labelpad=15)
	plt.legend()
	#plt.axis([0,T,0,1])
	plt.savefig('img/evolution_D-beta_' + str(int(beta_ref * 100)) +
		        '.pdf', bbox_inches='tight')


#plt.show()
