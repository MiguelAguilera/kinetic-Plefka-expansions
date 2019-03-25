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

size = 50
beta = 0.5
beta = 1.5
beta = 1
R = 10000
mode = 'c'
# mode='a'


T = 32
filename = 'data/m-' + mode + '-s-' + \
    str(size) + '-R-' + str(R) + '-beta-' + str(beta) + '.npz'
data = np.load(filename)
H = data['H']
J = data['J']
m_exp = data['m'][:, T]
C_exp = data['C'][:, :, T]
iu1 = np.triu_indices(size, 1)


print('TAP')
I = mf_ising(size)
I.H = H.copy()
I.J = J.copy()

I.initialize_state()
for t in range(T):
    I.update_P0_o2()
mP0o2 = I.m.copy()
CP0o2 = I.C.copy()

I.initialize_state()
for t in range(T):
    I.update_P0_o2_()
mP0o2_ = I.m.copy()
CP0o2_ = I.C.copy()

I.initialize_state()
for t in range(T):
    I.update_P1_o2()
mP2o2 = I.m.copy()
CP2o2 = I.C.copy()

I.initialize_state()
for t in range(T):
    I.update_P1_o2_()
mP2o2_ = I.m.copy()
CP2o2_ = I.C.copy()

plt.figure()
# plt.plot(sorted(m_exp),sorted(m_exp),'k.-',lw=1)
plt.plot(mP0o2 - m_exp, 'b.')
plt.plot(mP0o2_ - m_exp, 'r.')
plt.xlabel(r'$i$')
plt.ylabel(r'$m_{TAP}-m$')
plt.title(r'Plefka[t-1:t], $\beta=' + str(beta) + r'$')
plt.savefig(
    './img/m_Plefka[t-1:t]_beta=' +
    str(beta) +
    '.png',
    bbox_inches='tight')
# plt.figure()
# plt.plot(m_exp,mP0o2-m_exp,'b*')
# plt.plot(m_exp,mP0o2_-m_exp,'r.')


plt.figure()
# plt.plot(sorted(C_exp[iu1]),sorted(C_exp[iu1]),'k.-',lw=1)
plt.plot(CP0o2[iu1] - C_exp[iu1], 'b.')
plt.plot(CP0o2_[iu1] - C_exp[iu1], 'r.')
plt.xlabel(r'$I=iN+j$')
plt.ylabel(r'$C_{TAP}-C$')
plt.title(r'Plefka[t-1:t], $\beta=' + str(beta) + r'$')
plt.savefig(
    './img/C_Plefka[t-1:t]_beta=' +
    str(beta) +
    '.png',
    bbox_inches='tight')

# plt.figure()
# plt.plot(sorted(m_exp),sorted(m_exp),'k.-',lw=1)
# plt.plot(m_exp,mP2o2,'b*')
# plt.plot(m_exp,mP2o2_,'r.')
# plt.xlabel('C_{exp}')
# plt.ylabel(r'$C_{TAP}$')

plt.figure()
# plt.plot(sorted(m_exp),sorted(m_exp),'k.-',lw=1)
plt.plot(mP2o2 - m_exp, 'b.')
plt.plot(mP2o2_ - m_exp, 'r.')
plt.xlabel(r'$i$')
plt.ylabel(r'$m_{TAP}-m$')
plt.title(r'Plefka[t], $\beta=' + str(beta) + r'$')
plt.savefig(
    './img/m_Plefka[t]_beta=' +
    str(beta) +
    '.png',
    bbox_inches='tight')

plt.figure()
# plt.plot(sorted(C_exp[iu1]),sorted(C_exp[iu1]),'k.-',lw=1)
plt.plot(CP2o2[iu1] - C_exp[iu1], 'b.')
plt.plot(CP2o2_[iu1] - C_exp[iu1], 'r.')
plt.xlabel(r'$I=iN+j$')
plt.ylabel(r'$C_{TAP}-C$')
plt.title(r'Plefka[t], $\beta=' + str(beta) + r'$')
plt.savefig(
    './img/C_Plefka[t]_beta=' +
    str(beta) +
    '.png',
    bbox_inches='tight')

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
