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

plt.rc('text', usetex=True)
font = {'size': 15}
plt.rc('font', **font)
plt.rc('legend', **{'fontsize': 12})

size=512
beta = 1.0
R = 500000
gamma1 = 0.5
gamma2 = 0.1
random_s0 = False
stationary = False


#labels = [r'P[$t-1,t$]', r'P[$t$]', r'P[$t-1$]',r'P[$\mathbf{C}_t,\mathbf{D}_t$]']
labels = [r'P[$t-1,t$]', r'P[$t$]', r'P[$t-1$]',r'P2[$t$]']

T = 128
filename = 'data/m-c-ts0-gamma1-' + str(gamma1) +'-gamma2-' + str(gamma2) + '-s-' + \
            str(size) + '-R-' + str(R) + '-beta-' + str(beta) + '.npz'
data = np.load(filename)
H = data['H']
J = data['J']
m_exp = data['m'][:, T-1]
C_exp = data['C'][:, :, T-1]
D_exp = data['D'][:, :, T-2]
iu1 = np.triu_indices(size, 1)
s0 = np.array(data['s0'])

if random_s0:
    s0 = None
    print(s0)
else:
    print(np.mean(s0))
print('TAP')

I = mf_ising(size)
I.H = H.copy()
I.J = J.copy()

print('mP0o2')
I.initialize_state(s0)
for t in range(T):
    I.update_P0_o2()
mP0o2 = I.m.copy()
CP0o2 = I.C.copy()
DP0o2 = I.D.copy()

print('mP1o2')
I.initialize_state(s0)
for t in range(T):
    I.update_P1_o2()
mP1o2 = I.m.copy()
CP1o2 = I.C.copy()
DP1o2 = I.D.copy()

print('mP1Co2')
I.initialize_state(s0)
for t in range(T):
    I.update_P1C_o2()
mP1Co2 = I.m.copy()
CP1Co2 = I.C.copy()
DP1Co2 = I.D.copy()

print('mP2o1')
I.initialize_state(s0)
for t in range(T):
    I.update_P2_o1()
mP2o1 = I.m.copy()
CP2o1 = I.C.copy()
DP2o1 = I.D.copy()

cmap = cm.get_cmap('inferno_r')
colors=[]
for i in range(4):
	colors+=[cmap((i+0.5)/4)]
plt.figure(figsize=(5, 4),dpi=300)
plt.plot(sorted(m_exp), sorted(m_exp), 'k.-', lw=1)
plt.plot(m_exp, mP0o2, 'v',color=colors[0], ms=5, label=labels[0], rasterized=True)
plt.plot(m_exp, mP1o2, 's',color=colors[1], ms=5,  label=labels[1], rasterized=True)
plt.plot(m_exp, mP2o1, 'd',color=colors[2], ms=5,  label=labels[2], rasterized=True)
plt.plot(m_exp, mP1Co2, 'o',color=colors[3], ms=5,  label=labels[3], rasterized=True)
plt.plot([np.min(m_exp),np.max(m_exp)],[np.min(m_exp),np.max(m_exp)],'k')
plt.xlabel(r'$m_i^r$', fontsize=18)
plt.ylabel(r'$m_i^m$', fontsize=18, rotation=0, labelpad=15)
plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
plt.legend()
plt.savefig('img/distribution_m-beta_' +
            str(int(beta * 100)) + '.pdf', bbox_inches='tight')


plt.figure(figsize=(5, 4),dpi=300)
plt.plot(C_exp[iu1], CP0o2[iu1], 'v',color=colors[0], ms=5,  label=labels[0], rasterized=True)
plt.plot(C_exp[iu1], CP1o2[iu1], 's',color=colors[1], ms=5,  label=labels[1], rasterized=True)
plt.plot(C_exp[iu1], CP2o1[iu1], 'd',color=colors[2], ms=5,  label=labels[2], rasterized=True)
plt.plot(C_exp[iu1], CP1Co2[iu1], 'o',color=colors[3], ms=5,  label=labels[3], rasterized=True)
plt.plot([np.min(C_exp[iu1]),np.max(C_exp[iu1])],[np.min(C_exp[iu1]),np.max(C_exp[iu1])],'k')
plt.legend()
plt.xlabel(r'$C_{ij}^r$', fontsize=18)
plt.ylabel(r'$C_{ij}^m$', fontsize=18, rotation=0, labelpad=15)
plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
plt.savefig('img/distribution_C-beta_' +
            str(int(beta * 100)) + '.pdf', bbox_inches='tight')

plt.figure(figsize=(5, 4),dpi=300)
plt.plot(D_exp.flatten(), DP0o2.flatten(), 'v',color=colors[0], ms=5,  label=labels[0], rasterized=True)
plt.plot(D_exp.flatten(), DP1o2.flatten(), 's',color=colors[1], ms=5,  label=labels[1], rasterized=True)
plt.plot(D_exp.flatten(), DP2o1.flatten(), 'd',color=colors[2], ms=5,  label=labels[2], rasterized=True)
plt.plot(D_exp.flatten(), DP1Co2.flatten(), 'o',color=colors[3], ms=5, label=labels[3], rasterized=True)
plt.plot([np.min(D_exp),np.max(D_exp)],[np.min(D_exp),np.max(D_exp)],'k')
plt.legend()
plt.xlabel(r'$D_{ij}^r$', fontsize=18)
plt.ylabel(r'$D_{ij}^m$', fontsize=18, rotation=0, labelpad=15)
plt.title(r'$\beta/\beta_c=' + str(beta) + r'$', fontsize=18)
plt.savefig('img/distribution_D-beta_' +
            str(int(beta * 100)) + '.pdf', bbox_inches='tight')

#plt.show()

#print(np.sqrt(np.sum((C_exp-CP0o2)**2)), np.sqrt(np.sum((C_exp-CP1o2)**2)))

#

# print()
#print('TAP - Roudi')
#I1 = mf_ising_roudi(size)
#I1.H = H.copy()
#I1.J = J.copy()
#
# I1.initialize_state(m_exp)
# for t in range(T):
#	I1.update_TAP()
#	I1.update_correlations()
#	I1.update_delayed_correlations()
# print(I1.m)
#
# plt.figure()
# plt.plot(m_exp,I1.m,'+')
#
##
#print(np.sqrt(np.sum((m_exp-I.m)**2)), np.sqrt(np.sum((m_exp-I1.m)**2)))
#
# plt.figure()
# plt.plot(C_exp[iu1],I.C[iu1],'*')
# plt.plot(C_exp[iu1],I1.C[iu1],'+')
#
# plt.plot(sorted(C_exp[iu1]),sorted(C_exp[iu1]),'k.-',lw=1)
#
#
#print(np.sqrt(np.sum((C_exp-I.C)**2)), np.sqrt(np.sum((C_exp-I1.C)**2)))
