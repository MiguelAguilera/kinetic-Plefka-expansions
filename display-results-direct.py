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

import matplotlib as mpl
print(mpl.rcParams['agg.path.chunksize'])
mpl.rcParams['agg.path.chunksize'] = 10000
print(mpl.rcParams['agg.path.chunksize'])
#exit()

plt.rc('text', usetex=True)
font = {'size': 15}
plt.rc('font', **font)
plt.rc('legend', **{'fontsize': 18})


# size = 200


#size = 16
#size= 32
#size = 64
#size = 128
size = 256
size=512
beta = 0.7
beta = 1.0
#beta = 1.3
R = 10000
mode = 'c'
gamma1 = 0.5
gamma2 = 0.1
#gamma1 = 1.0
random_s0 = False
# stationary=True
stationary = False

# random_s0 = True

T = 128

cmap = cm.get_cmap('plasma_r')
colors=[]
for i in range(4):
	colors+=[cmap((i)/3)]
#cmap = cm.get_cmap('inferno_r')
#colors=[]
#for i in range(4):
#	colors+=[cmap((i+0.5)/4)]
line=['--','--','-','-']
line=[(5, 4),(5, 4),'','']

labels = [r'Plefka[$t-1,t$]', r'Plefka[$t$]', r'Plefka[$t-1$]',r'Plefka2[$t$]',r'Original']
lws = [2,2,2,3,1.5]

filename='img/compare-T_100_size_'+str(size)+'.npz'
data=np.load(filename)
m_exp=data['m_exp']
C_exp=data['C_exp']
D_exp=data['D_exp']

mP0o2_mean=data['mP0o2_mean']
mP1o2_mean=data['mP1o2_mean']
mP2o1_mean=data['mP2o1_mean']
mP1Co2_mean=data['mP1Co2_mean']
CP0o2_mean=data['CP0o2_mean']
CP1o2_mean=data['CP1o2_mean']
CP2o1_mean=data['CP2o1_mean']
CP1Co2_mean=data['CP1Co2_mean']
DP0o2_mean=data['DP0o2_mean']
DP1o2_mean=data['DP1o2_mean']
DP2o1_mean=data['DP2o1_mean']
DP1Co2_mean=data['DP1Co2_mean']

mPexp_mean=data['mPexp_mean']
CPexp_mean=data['CPexp_mean']
DPexp_mean=data['DPexp_mean']

mP0o2=data['mP0o2']
mP1o2=data['mP1o2']
mP2o1=data['mP2o1']
mP1Co2=data['mP1Co2']
CP0o2=data['CP0o2']
CP1o2=data['CP1o2']
CP2o1=data['CP2o1']
CP1Co2=data['CP1Co2']
DP0o2=data['DP0o2']
DP1o2=data['DP1o2']
DP2o1=data['DP2o1']
DP1Co2=data['DP1Co2']

EmP0o2=data['EmP0o2']
EmP1o2=data['EmP1o2']
EmP2o1=data['EmP2o1']
EmP1Co2=data['EmP1Co2']
ECP0o2=data['ECP0o2']
ECP1o2=data['ECP1o2']
ECP2o1=data['ECP2o1']
ECP1Co2=data['ECP1Co2']
EDP0o2=data['EDP0o2']
EDP1o2=data['EDP1o2']
EDP2o1=data['EDP2o1']
EDP1Co2=data['EDP1Co2']

steps = np.arange(T + 1)
iu1 = np.triu_indices(size, 1)

B=11
betas = 1 + np.linspace(-1,1,B)*0.3


M=4
N=len(betas)
errorm=np.zeros((M,N))
errorC=np.zeros((M,N))
errorD=np.zeros((M,N))
if gamma1==0.5 and gamma2==0.1:
	beta0 = 1.1123

for ind in range(len(betas)):
	beta_ref = round(betas[ind],3)
	beta = beta_ref * beta0
#	beta=betas[ind]
	filename='img/compare-T_' + str(int(beta_ref * 100)) + '_size_'+str(size)+'.npz'
	data=np.load(filename)
	print(list(data.keys()))
#	exit()

	if beta==1:
		mP0o2=data['mP0o2']
		mP1o2=data['mP1o2']
		mP2o1=data['mP2o1']
		mP1Co2=data['mP1Co2']
		CP0o2=data['CP0o2']
		CP1o2=data['CP1o2']
		CP2o1=data['CP2o1']
		CP1Co2=data['CP1Co2']
		DP0o2=data['DP0o2']
		DP1o2=data['DP1o2']
		DP2o1=data['DP2o1']
		DP1Co2=data['DP1Co2']
	
	EmP0o2=np.mean(data['EmP0o2'])
	EmP1o2=np.mean(data['EmP1o2'])
	EmP2o1=np.mean(data['EmP2o1'])
	EmP1Co2=np.mean(data['EmP1Co2'])
	ECP0o2=np.mean(data['ECP0o2'])
	ECP1o2=np.mean(data['ECP1o2'])
	ECP2o1=np.mean(data['ECP2o1'])
	ECP1Co2=np.mean(data['ECP1Co2'])
	EDP0o2=np.mean(data['EDP0o2'])
	EDP1o2=np.mean(data['EDP1o2'])
	EDP2o1=np.mean(data['EDP2o1'])
	EDP1Co2=np.mean(data['EDP1Co2'])

	names = ['P[t-1:t]', 'P[t-1]', 'P[t]','P[D]']
	x = np.linspace(-0.35,0.35,4)
	errorm[:,ind]=np.array([EmP0o2,EmP1o2,EmP2o1,EmP1Co2])
	errorC[:,ind]=np.array([ECP0o2,ECP1o2,ECP2o1,ECP1Co2])
	errorD[:,ind]=np.array([EDP0o2,EDP1o2,EDP2o1,EDP1Co2])

	names = ['P[t-1:t]', 'P[t-1]', 'P[t]','P[D]']
	x = np.linspace(-0.35,0.35,4)
#	errorm=np.array([np.mean((m_exp-mP0o2)**2), np.mean((m_exp-mP2o1)**2), np.mean((m_exp-mP1o2)**2), np.mean((m_exp-mP1Co2)**2)])
	errorm[:,ind]=np.array([EmP0o2,EmP1o2,EmP2o1,EmP1Co2])
	print(errorm)
#	errorC=np.array([np.mean((C_exp[iu1].flatten()-CP0o2[iu1].flatten())**2), np.mean((C_exp[iu1].flatten()-CP2o1[iu1].flatten())**2), np.mean((C_exp[iu1].flatten()-CP1o2[iu1].flatten())**2), np.mean((C_exp[iu1].flatten()-CP1Co2[iu1].flatten())**2)]  )
	errorC[:,ind]=np.array([ECP0o2,ECP1o2,ECP2o1,ECP1Co2])
	print(errorC)
#	errorD=np.array([np.mean((D_exp.flatten()-DP0o2.flatten())**2), np.mean((D_exp.flatten()-DP2o1.flatten())**2), np.mean((D_exp.flatten()-DP1o2.flatten())**2), np.mean((D_exp.flatten()-DP1Co2.flatten())**2)]  )
	errorD[:,ind]=np.array([EDP0o2,EDP1o2,EDP2o1,EDP1Co2])
	print(errorD)



letters = ['A','B','C','D','E','F','G','H','I']
pos_l = [-0.2, 1.0]
#fig, ax = plt.subplots(3, 3, figsize=(15, 12))

#fig, ax = plt.subplots(3, 3, figsize=(16, 8),dpi=300)
fig, ax = plt.subplots(3, 3, figsize=(16, 10),dpi=300)
#plt.subplot(331)
ax[0, 0].plot(steps[0:1], mPexp_mean[0:1], color='k', label=labels[4])
ax[0, 0].plot(steps, mP0o2_mean, dashes=line[0],color=colors[0],lw=lws[0], label=labels[0])
ax[0, 0].plot(steps, mP1o2_mean, dashes=line[1],color=colors[1],lw=lws[1], label=labels[1])
ax[0, 0].plot(steps, mP2o1_mean, dashes=line[2],color=colors[2],lw=lws[2], label=labels[2])
ax[0, 0].plot(steps, mP1Co2_mean, dashes=line[3],color=colors[3],lw=lws[3], label=labels[3])
ax[0, 0].plot(steps, mP0o2_mean, dashes=line[0],color=colors[0],lw=lws[0])
ax[0, 0].plot(steps, mP1o2_mean, dashes=line[1],color=colors[1],lw=lws[1])
ax[0, 0].plot(steps[0:1], mPexp_mean[0:1], color='k')
ax[0, 0].plot(steps, mPexp_mean, 'k',lw=lws[4])#, label=r'$P$')
#ax[0, 0].set_title(r'$\beta/\beta_c=' + str(1) + r'$', fontsize=18)
ax[0, 0].set_xlabel(r'$t$', fontsize=18)
ax[0, 0].set_ylabel(r'$\langle m_{i,t} \rangle$', fontsize=18, rotation=0, labelpad=25)
#ax[0, 0].legend(loc="upper right",ncol=2)
ax[0, 0].text(pos_l[0],pos_l[1], r'\textbf '+letters[0], transform=ax[0, 0].transAxes,
      fontsize=20, va='top', ha='right')
ax[0, 0].axis([0,T,0.2,1])
#plt.show()
#plt.savefig('img/evolution_m-beta_' + str(int(beta * 100)) +
#	        '.pdf', bbox_inches='tight')


#fig, ax = plt.subplots(1, 1, figsize=(5, 4))
#plt.subplot(332)

ax[0, 1].plot(steps, CP2o1_mean, dashes=line[2],color=colors[2],lw=lws[2])#, label=labels[2])
ax[0, 1].plot(steps, CP1Co2_mean, dashes=line[3],color=colors[3],lw=lws[3])#, label=labels[3])

ax[0, 1].plot(steps, CP0o2_mean, dashes=line[0],color=colors[0],lw=lws[0])#, label=labels[0])
ax[0, 1].plot(steps, CP1o2_mean, dashes=line[1],color=colors[1],lw=lws[1])#, label=labels[1])

ax[0, 1].plot(steps, CPexp_mean, 'k',lw=lws[4])#, label=r'$P$')
#ax[0, 1].set_title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
ax[0, 1].set_xlabel(r'$t$', fontsize=18)
ax[0, 1].set_ylabel(r'$\langle C_{ik,t} \rangle$', fontsize=18, rotation=0, labelpad=25)
ax[0, 1].set_yticks([0.00,0.01])
#ax[0, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[0, 1].text(pos_l[0],pos_l[1], r'\textbf '+letters[1], transform=ax[0, 1].transAxes,
      fontsize=20, va='top', ha='right')
#plt.legend(loc='lower right')
ax[0, 1].axis([0,T,0,0.017])
#plt.savefig('img/evolution_C-beta_' + str(int(beta * 100)) +
#	        '.pdf', bbox_inches='tight')

#fig, ax = plt.subplots(1, 1, figsize=(5, 4))
#plt.subplot(333)

ax[0, 2].plot(steps, DP2o1_mean, dashes=line[2],color=colors[2],lw=lws[2])#, label=labels[2])
ax[0, 2].plot(steps, DP1Co2_mean, dashes=line[3],color=colors[3],lw=lws[3])#, label=labels[3])

ax[0, 2].plot(steps, DP0o2_mean, dashes=line[0],color=colors[0],lw=lws[0])#, label=labels[0])
ax[0, 2].plot(steps, DP1o2_mean, dashes=line[1],color=colors[1],lw=lws[1])#, label=labels[1])

ax[0, 2].plot(steps, DPexp_mean, 'k',lw=lws[4])#, label=r'$P$')
#ax[0, 2].set_title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
ax[0, 2].set_yticks([0.00,0.01])
ax[0, 2].set_xlabel(r'$t$', fontsize=18)
ax[0, 2].set_ylabel(r'$\langle D_{il,t} \rangle$', fontsize=18, rotation=0, labelpad=25)
ax[0, 2].text(pos_l[0],pos_l[1], r'\textbf '+letters[2], transform=ax[0, 2].transAxes,
      fontsize=20, va='top', ha='right')
ax[0, 2].axis([0,T,0,0.017])
#plt.legend(loc='lower right')
#plt.axis([0,T,0,1])
#plt.savefig('img/evolution_D-beta_' + str(int(beta * 100)) +
#	        '.pdf', bbox_inches='tight')



#fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax[1, 0].plot(m_exp, mP0o2, 'v', color=colors[0], ms=3, rasterized=True)# label=labels[0]
ax[1, 0].plot(m_exp, mP1o2, 's',  color=colors[1], ms=3, rasterized=True)#, label=labels[1], rasterized=True)
ax[1, 0].plot(m_exp, mP2o1, 'd',  color=colors[2], ms=3, rasterized=True)#, label=labels[2], rasterized=True)
ax[1, 0].plot(m_exp, mP1Co2, 'o',  color=colors[3], ms=3, rasterized=True)#, label=labels[3], rasterized=True)
ax[1, 0].plot(sorted(m_exp), sorted(m_exp), 'k',lw=2.5, rasterized=True)
ax[1, 0].set_xlabel(r'$\mathbf{m}^o$', fontsize=18)
ax[1, 0].set_ylabel(r'$\mathbf{m}^p$', fontsize=18, rotation=0, labelpad=15)
#plt.title(r'$\beta=' + str(beta) + r'$', fontsize=18)
#ax[1, 0].legend(loc="lower right")
ax[1, 0].text(pos_l[0],pos_l[1], r'\textbf '+letters[3], transform=ax[1, 0].transAxes,
      fontsize=20, va='top', ha='right')

#plt.savefig('img/distribution_m-beta_' +
#            str(int(beta * 100)) + '.pdf', bbox_inches='tight')

#fig, ax = plt.subplots(1, 1, figsize=(5, 4))

ax[1, 1].plot(C_exp[iu1], CP0o2[iu1], 'v', color=colors[0], ms=3, rasterized=True)#, label=labels[0], rasterized=True)
ax[1, 1].plot(C_exp[iu1], CP1o2[iu1], 's', color=colors[1], ms=3, rasterized=True)#, label=labels[1], rasterized=True)
ax[1, 1].plot(C_exp[iu1], CP2o1[iu1], 'd', color=colors[2], ms=3, rasterized=True)#, label=labels[2], rasterized=True)
ax[1, 1].plot(C_exp[iu1], CP1Co2[iu1], 'o', color=colors[3], ms=3, rasterized=True)#, label=labels[3], rasterized=True)
ax[1, 1].plot(sorted(C_exp[iu1]), sorted(C_exp[iu1]), 'k', lw=2.5, rasterized=True)
#ax[1, 1].legend()
ax[1, 1].set_xlabel(r'$\mathbf{C}^o$', fontsize=18)
ax[1, 1].set_ylabel(r'$\mathbf{C}^p$', fontsize=18, rotation=0, labelpad=15)
ax[1, 1].text(pos_l[0],pos_l[1], r'\textbf '+letters[4], transform=ax[1, 1].transAxes,
      fontsize=20, va='top', ha='right')
#plt.title(r'$\beta=' + str(beta) + r'$', fontsize=18)
#plt.savefig('img/distribution_C-beta_' +
#            str(int(beta * 100)) + '.pdf', bbox_inches='tight')

#fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax[1, 2].plot(D_exp.flatten(), DP0o2.flatten(), 'v',  color=colors[0], ms=3, rasterized=True)#, label=labels[0], rasterized=True)
ax[1, 2].plot(D_exp.flatten(), DP1o2.flatten(), 's',  color=colors[1], ms=3, rasterized=True)#, label=labels[1], rasterized=True)
ax[1, 2].plot(D_exp.flatten(), DP2o1.flatten(), 'd',  color=colors[2], ms=3, rasterized=True)#, label=labels[2], rasterized=True)
ax[1, 2].plot(D_exp.flatten(), DP1Co2.flatten(), 'o',  color=colors[3], ms=3, rasterized=True)#, label=labels[3], rasterized=True)
ax[1, 2].plot(sorted(D_exp.flatten()), sorted(D_exp.flatten()), 'k', lw=2.5, rasterized=True)
#ax[1, 2].legend(loc=3)
ax[1, 2].set_xlabel(r'$\mathbf{D}^o$', fontsize=18)
ax[1, 2].set_ylabel(r'$\mathbf{D}^p$', fontsize=18, rotation=0, labelpad=15)
ax[1, 2].text(pos_l[0],pos_l[1], r'\textbf '+letters[5], transform=ax[1, 2].transAxes,
      fontsize=20, va='top', ha='right')
#plt.title(r'$\beta=' + str(beta) + r'$', fontsize=18)
#plt.savefig('img/distribution_D-beta_' +
#            str(int(beta * 100)) + '.pdf', bbox_inches='tight')




#fig, ax = plt.subplots(1, 1, figsize=(5, 4))
#plt.subplot(337)
for m in range(M):
	ax[2, 0].semilogy(betas,errorm[m,:],dashes=line[m],color=colors[m],lw=lws[m])#, label=labels[m])
for m in range(2):
	ax[2, 0].semilogy(betas,errorm[m,:],dashes=line[m],color=colors[m],lw=lws[m])
ax[2, 0].set_xlabel(r'$\beta/\beta_c$', fontsize=18)
ax[2, 0].set_ylabel(r'$\epsilon_m$', fontsize=18, rotation=0, labelpad=15)
#ax[2, 0].legend(bbox_to_anchor=(0.55, 0.5))
ax[2, 0].text(pos_l[0],pos_l[1], r'\textbf '+letters[6], transform=ax[2, 0].transAxes,
      fontsize=20, va='top', ha='right')
#plt.savefig('img/error_m.pdf', bbox_inches='tight')

#fig, ax = plt.subplots(1, 1, figsize=(5, 4))
#plt.subplot(338)
for m in range(M):
	ax[2, 1].semilogy(betas,errorC[m,:],dashes=line[m],color=colors[m],lw=lws[m])#, label=labels[m])
for m in range(2):
	ax[2, 1].semilogy(betas,errorC[m,:],dashes=line[m],color=colors[m],lw=lws[m])
ax[2, 1].set_xlabel(r'$\beta/\beta_c$', fontsize=18)
ax[2, 1].set_ylabel(r'$\epsilon_C$', fontsize=18, rotation=0, labelpad=15)
ax[2, 1].text(pos_l[0],pos_l[1], r'\textbf '+letters[7], transform=ax[2, 1].transAxes,
      fontsize=20, va='top', ha='right')
#ax[0].legend(loc=3)
#plt.savefig('img/error_C.pdf', bbox_inches='tight')


#fig, ax = plt.subplots(1, 1, figsize=(5, 4))
#plt.subplot(339)
for m in range(M):
	ax[2, 2].semilogy(betas,errorD[m,:],dashes=line[m],color=colors[m],lw=lws[m])#, label=labels[m])
for m in range(2):
	ax[2, 2].semilogy(betas,errorD[m,:],dashes=line[m],color=colors[m],lw=lws[m])
ax[2, 2].set_xlabel(r'$\beta/\beta_c$', fontsize=18)
ax[2, 2].set_ylabel(r'$\epsilon_D$', fontsize=18, rotation=0, labelpad=15)
ax[2, 2].text(pos_l[0],pos_l[1], r'\textbf '+letters[8], transform=ax[2, 2].transAxes,
      fontsize=20, va='top', ha='right')
#ax[2, 2].legend()
#plt.savefig('img/error_D.pdf', bbox_inches='tight')
plt.figlegend(loc='upper center',bbox_to_anchor=(0.5, 1.), borderaxespad=0.1, ncol=5 )

#fig.legend()
plt.figlegend(loc='upper center',bbox_to_anchor=(0.5, 1.), borderaxespad=0, ncol=5 )
fig.tight_layout(h_pad=0.3,w_pad=0.7,rect=[0,0,1,0.975])
plt.savefig('img/results-direct-Ising.pdf', bbox_inches='tight')


