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
plt.rc('legend', **{'fontsize': 18})

# size = 200


#size = 16
#size= 32
#size = 64
#size = 128
size = 256
size=512
#beta = 0.7
beta = 1.0
#beta = 1.3
R = 100000
mode = 'c'
gamma1 = 0.5
gamma2 = 0.1
#gamma1 = 1.0
random_s0 = False
# stationary=True
stationary = False

# random_s0 = True

T = 64
iu1 = np.triu_indices(size, 1)


#figm, axm = plt.subplots(1, 1, figsize=(5, 4))
#figC, axC = plt.subplots(1, 1, figsize=(5, 4))
#figD, axD = plt.subplots(1, 1, figsize=(5, 4))

#figH, axH = plt.subplots(1, 1, figsize=(5, 4))
#figJ, axJ = plt.subplots(1, 1, figsize=(5, 4))

betas=np.array([0.7,0.8,0.9,1.0,1.1,1.2,1.3])
offset=np.arange(3)
width=0.20

beta=1.0

if gamma1==0.5 and gamma2==0.1:
	beta0 = 1.1123
B=11
betas = 1 + np.linspace(-1,1,B)*0.3
#for beta in [0.7,0.8,0.9,1.0,1.1,1.2,1.3]:

N=len(betas)
M=4
errorm=np.zeros((M,N))
errorC=np.zeros((M,N))
errorD=np.zeros((M,N))
errorH=np.zeros((M,N))
errorJ=np.zeros((M,N))
if gamma1==0.5 and gamma2==0.1:
	beta0 = 1.1123
B=11
betas = 1 + np.linspace(-1,1,B)*0.3
#for beta in [0.7,0.8,0.9,1.0,1.1,1.2,1.3]:
for ind in range(len(betas)):
	beta_ref = round(betas[ind],3)
	beta = beta_ref * beta0
#	beta=betas[ind]
	filename='img/compare-T_' + str(int(beta_ref * 100)) + '_size_'+str(size)+'.npz'
	data=np.load(filename)
#	m_exp=data['m_exp']
#	C_exp=data['C_exp']
#	D_exp=data['D_exp']

#	EmP0o2=data['EmP0o2'][T-1]
#	EmP1o2=data['EmP1o2'][T-1]
#	EmP2o1=data['EmP2o1'][T-1]
#	EmP1Co2=data['EmP1Co2'][T-1]
#	ECP0o2=data['ECP0o2'][T-1]
#	ECP1o2=data['ECP1o2'][T-1]
#	ECP2o1=data['ECP2o1'][T-1]
#	ECP1Co2=data['ECP1Co2'][T-1]
#	EDP0o2=data['EDP0o2'][T-1]
#	EDP1o2=data['EDP1o2'][T-1]
#	EDP2o1=data['EDP2o1'][T-1]
#	EDP1Co2=data['EDP1Co2'][T-1]

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

#	filename='img/compare-J_' + str(int(beta_ref * 100)) + '.npz'
#	data=np.load(filename)
#	errorH[:,ind]=data['errorH']
#	errorJ[:,ind]=data['errorJ']
	filename='img/compare-J_' + str(int(beta_ref * 100)) +'.npz'
	data=np.load(filename)
	HP0o2=data['HP0o2']
	HP1o2=data['HP1o2']
	HP2o1=data['HP2o1']
	HP1Co2=data['HP1Co2']
	JP0o2=data['JP0o2']
	JP1o2=data['JP1o2']
	JP2o1=data['JP2o1']
	JP1Co2=data['JP1Co2']
	H=data['H']
	J=data['J']
	errorH[:,ind]=np.array([np.mean((H-HP0o2)**2), np.mean((H-HP1o2)**2), np.mean((H-HP2o1)**2), np.mean((H-HP1Co2)**2)])
	errorJ[:,ind]=np.array([np.mean((J.flatten()-JP0o2.flatten())**2), np.mean((J.flatten()-JP1o2.flatten())**2), np.mean((J.flatten()-JP2o1.flatten())**2), np.mean((J.flatten()-JP1Co2.flatten())**2)]  )


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
	
cmap = cm.get_cmap('plasma_r')
colors=[]
for i in range(4):
	colors+=[cmap((i)/3)]
#cmap = cm.get_cmap('inferno_r')
#colors=[]
#for i in range(4):
#	colors+=[cmap((i+0.5)/4)]
#line=['--','--','-','-']
line=[(5, 4),(5, 4),'','']

labels = [r'Plefka[$t-1,t$]', r'Plefka[$t$]', r'Plefka[$t-1$]',r'Plefka2[$t$]',r'Original']
lws = [2,2,2,3,1.5]


#fig, ax = plt.subplots(1, 1, figsize=(5, 4))
#for m in range(M):
#	plt.semilogy(betas,errorm[m,:],line[m],color=colors[m],lw=lws[m], label=labels[m])
#for m in range(2):
#	plt.semilogy(betas,errorm[m,:],line[m],color=colors[m],lw=lws[m])
#plt.xlabel(r'$\beta/\beta_c$', fontsize=18)
#plt.ylabel(r'$\epsilon_m$', fontsize=18, rotation=0, labelpad=15)
#plt.legend()
##plt.axis([0,T,0,1])
#plt.savefig('img/error_m.pdf', bbox_inches='tight')


#fig, ax = plt.subplots(1, 1, figsize=(5, 4))
#for m in range(M):
#	plt.semilogy(betas,errorC[m,:],line[m],color=colors[m],lw=lws[m], label=labels[m])
#for m in range(2):
#	plt.semilogy(betas,errorC[m,:],line[m],color=colors[m],lw=lws[m])
#plt.xlabel(r'$\beta/\beta_c$', fontsize=18)
#plt.ylabel(r'$\epsilon_C$', fontsize=18, rotation=0, labelpad=15)
#plt.legend(loc=3)
##plt.axis([0,T,0,1])
#plt.savefig('img/error_C.pdf', bbox_inches='tight')


#fig, ax = plt.subplots(1, 1, figsize=(5, 4))
#for m in range(M):
#	plt.semilogy(betas,errorD[m,:],line[m],color=colors[m],lw=lws[m], label=labels[m])
#for m in range(2):
#	plt.semilogy(betas,errorD[m,:],line[m],color=colors[m],lw=lws[m])
#plt.xlabel(r'$\beta/\beta_c$', fontsize=18)
#plt.ylabel(r'$\epsilon_D$', fontsize=18, rotation=0, labelpad=15)
#plt.legend()
##plt.axis([0,T,0,1])
#plt.savefig('img/error_D.pdf', bbox_inches='tight')

##plt.show()

beta=1
filename='img/compare-J_100.npz'
data=np.load(filename)
HP0o2=data['HP0o2']
HP1o2=data['HP1o2']
HP2o1=data['HP2o1']
HP1Co2=data['HP1Co2']
JP0o2=data['JP0o2']
JP1o2=data['JP1o2']
JP2o1=data['JP2o1']
JP1Co2=data['JP1Co2']
H=data['H']
J=data['J']



letters = ['A','B','C','D']
pos_l = [-0.2*2/3, 1.0]
#fig, ax = plt.subplots(3, 3, figsize=(15, 12))
fig, ax = plt.subplots(2, 2, figsize=(16*2/3, 10*2/3),dpi=300)
#plt.subplot(331)
ax[0, 0].plot(H,HP0o2, 'v', color=colors[0], ms=5, rasterized=True)
ax[0, 0].plot(H,HP1o2, 's', color=colors[1], ms=5, rasterized=True)
ax[0, 0].plot(H,HP2o1, 'd', color=colors[2], ms=5, rasterized=True)
ax[0, 0].plot(H,HP1Co2, 'o', color=colors[3], ms=5, rasterized=True)
ax[0, 0].plot([np.min(H),np.max(H)],[np.min(H),np.max(H)],'k',lw=2.5)
ax[0, 0].axis([np.min(H),np.max(H),np.min(np.concatenate((HP0o2,HP1o2,HP1Co2))),np.max(np.concatenate((HP0o2,HP1o2,HP1Co2)))])
ax[0, 0].set_xlabel(r'$H_i^o$', fontsize=18)
ax[0, 0].set_ylabel(r'$H_i^p$', fontsize=18, rotation=0, labelpad=15)
#plt.title(r'$\beta=' + str(beta) + r'$', fontsize=18)
#ax[1, 0].legend(loc="lower right")
ax[0, 0].text(pos_l[0],pos_l[1], r'\textbf '+letters[0], transform=ax[0, 0].transAxes,
      fontsize=20, va='top', ha='right')


#ax[0, 1].plot(J[0],JP0o2[0], 'v', color=colors[0], ms=5, label=labels[0], rasterized=True)
ax[0, 1].plot(J.flatten(),JP0o2.flatten(), 'v', color=colors[0], ms=5, rasterized=True)
ax[0, 1].plot(J.flatten(),JP1o2.flatten(), 's', color=colors[1], ms=5, rasterized=True)
ax[0, 1].plot(J.flatten(),JP2o1.flatten(), 'd', color=colors[2], ms=5, rasterized=True)
ax[0, 1].plot(J.flatten(),JP1Co2.flatten(), 'o', color=colors[3], ms=5, rasterized=True)
ax[0, 1].plot([np.min(J),np.max(J)],[np.min(J),np.max(J)],'k',lw=2.5)
#ax[0, 1].axis([np.min(J),np.max(J),2*np.min(np.concatenate((JP0o2,JP1Co2))),2*np.max(np.concatenate((JP0o2,JP1Co2)))])
ax[0, 1].set_xlabel(r'$J_{ij}^o$', fontsize=18)
ax[0, 1].set_ylabel(r'$J_{ij}^p$', fontsize=18, rotation=0, labelpad=15)
ax[0, 1].text(pos_l[0],pos_l[1], r'\textbf '+letters[1], transform=ax[0, 1].transAxes,
      fontsize=20, va='top', ha='right')
ax[0, 1].axis([np.min(J),np.max(J),np.min(J),np.max(J)*1.6])
#plt.legend(loc='lower right')
#plt.savefig('img/evolution_C-beta_' + str(int(beta * 100)) +
#	        '.pdf', bbox_inches='tight')



#fig, ax = plt.subplots(1, 1, figsize=(5, 4))
#plt.subplot(337)
for m in range(M):
	ax[1, 0].semilogy(betas,errorH[m,:],dashes=line[m],color=colors[m],lw=lws[m])#, label=labels[m])
for m in range(2):
	ax[1, 0].semilogy(betas,errorH[m,:],dashes=line[m],color=colors[m],lw=lws[m])#, label=labels[m])
ax[1, 0].set_xlabel(r'$\beta/\beta_c$', fontsize=18)
ax[1, 0].set_ylabel(r'$\epsilon_H$', fontsize=18, rotation=0, labelpad=15)
#ax[1, 0].legend(loc='center left')
ax[1, 0].text(pos_l[0],pos_l[1], r'\textbf '+letters[2], transform=ax[1, 0].transAxes,
      fontsize=20, va='top', ha='right')
#plt.axis([0,T,0,1])
#plt.savefig('img/error_m.pdf', bbox_inches='tight')

#fig, ax = plt.subplots(1, 1, figsize=(5, 4))
#plt.subplot(338)
#ax[0, 0].plot(H[0:1],H[0:1], color='k', label=labels[4])
for m in range(M):
	ax[1, 1].semilogy(betas,errorJ[m,:],dashes=line[m],color=colors[m],lw=lws[m], label=labels[m])
for m in range(2):
	ax[1, 1].semilogy(betas,errorJ[m,:],dashes=line[m],color=colors[m],lw=lws[m])
ax[1, 1].set_xlabel(r'$\beta/\beta_c$', fontsize=18)
ax[1, 1].set_ylabel(r'$\epsilon_J$', fontsize=18, rotation=0, labelpad=15)

ax[1, 1].text(pos_l[0],pos_l[1], r'\textbf '+letters[3], transform=ax[1, 1].transAxes,
      fontsize=20, va='top', ha='right')
#ax[0].legend(loc=3)
#plt.axis([0,T,0,1])
#plt.savefig('img/error_C.pdf', bbox_inches='tight')

plt.figlegend(loc='upper center',bbox_to_anchor=(0.5, 1.), borderaxespad=0, ncol=5 )
fig.tight_layout(h_pad=0.3,w_pad=0.7,rect=[0,0,1,0.975])

plt.savefig('img/results-inverse-Ising.pdf', bbox_inches='tight')

