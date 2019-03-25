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

size = 100
beta = 1
#beta = 0.5
#beta = 1.5
R = 10000
mode = 'c'

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
    I.update_P1_o2()
mP1o2 = I.m.copy()
CP1o2 = I.C.copy()

I.initialize_state()
for t in range(T):
    I.update_P1C_o2()
mP1Co2 = I.m.copy()
CP1Co2 = I.C.copy()

I.initialize_state()
for t in range(T):
    I.update_P0_o2()
mP0o2 = I.m.copy()
CP0o2 = I.C.copy()

# I.initialize_state()
# for t in range(T):
#	I.update_P4_o1()
# mP4o1=I.m.copy()
# CP4o1=I.C.copy()

# I.initialize_state()
# for t in range(T):
#	I.update_P4_o2()
# mP4o2=I.m.copy()
# CP4o2=I.C.copy()
#
# I.initialize_state()
# for t in range(T):
#	I.update_P5_o2()
# mP5o2=I.m.copy()
# CP5o2=I.C.copy()

plt.figure()
plt.plot(sorted(m_exp), sorted(m_exp), 'k.-', lw=1)
plt.plot(m_exp,mP0o2,'b.')
plt.plot(m_exp, mP1o2, 'g.')
plt.plot(m_exp, mP1Co2, 'r.')
# plt.plot(m_exp,mP4o1,'go',ms=8)
# plt.plot(m_exp,mP4o2,'r*')
# plt.plot(m_exp,mP5o2,'g.')
plt.figure()
plt.plot(C_exp[iu1], CP0o2[iu1], 'b.')
plt.plot(C_exp[iu1],CP1o2[iu1],'g.')
plt.plot(C_exp[iu1], CP1Co2[iu1], 'r.')
# plt.plot(C_exp[iu1],CP4o1[iu1],'go')
# plt.plot(C_exp[iu1],CP4o2[iu1],'g*')
# plt.plot(C_exp[iu1],CP5o2[iu1],'g.')

plt.plot(sorted(C_exp[iu1]), sorted(C_exp[iu1]), 'k.-', lw=1)


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
