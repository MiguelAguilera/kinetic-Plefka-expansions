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

mode = 'a'
mode = 'c'
size = 100
I = ising(size)
#
# beta=0.9
# beta=0.5
beta = 1
gamma1 = 0.3
gamma2 = 0.3
I.H = -0.0 + gamma1 * np.random.randn(size)
if mode == 'c':
    I.J = beta * (1 + gamma2 * np.random.randn(size, size)) / size
else:
    I.J = 1.0 * np.random.randn(size, size) / size * 10
# I.h=np.array([-1,1,1])
# I.J=np.zeros((size,size))
#I.J = 2.*np.random.randn(size,size)/size**0.5
# for i in range(size):

J = I.J.copy()
H = I.H.copy()

T = 2**6
m_exp = np.zeros((size, T))
C_exp = np.zeros((size, size, T))
D_exp = np.zeros((size, size, T))

R = 10000
print('generate initial state')
I.randomize_state()
for t in range(size * 100):
    I.GlauberStep()
s0 = I.s.copy()

# s0=np.ones(size)
# s0[0:size//2]=-1
# s0[0:2]=-1
print('generate data')
for rep in range(R):
    print(rep, '/', R)
    I.s = s0.copy()
#	I.randomize_state()
    for t in range(T):
        s_prev = I.s.copy()
        I.GlauberStep()
        h = I.H + np.dot(I.J, s_prev)
        m_exp[:, t] += np.tanh(h) / R
        C_exp[:, :, t] += np.einsum('i,j->ij',
                                    np.tanh(h), np.tanh(h), optimize=True) / R
C_exp -= np.einsum('it,jt->ijt', m_exp, m_exp, optimize=True)

filename = 'data/m-' + mode + '-s-' + \
    str(size) + '-R-' + str(R) + '-beta-' + str(beta) + '.npz'
np.savez(filename, C=C_exp, m=m_exp, H=H, J=J, s0=s0)

# print(m_exp)
# print(C_exp)
##np.einsum('ii->i', C_exp, optimize=True)[:] = 1 - m_exp**2
# print(m_exp)
# print(min(m_exp),max(m_exp))
#
#
#iu1 = np.triu_indices(size, 1)
#
#
# print(C_exp)
#
# print('TAP')
#I = mf_ising(size)
#I.H = H.copy()
#I.J = J.copy()
#
# I.initialize_state()
# for t in range(T):
#	I.update_P2_o2()
# mP2o2=I.m.copy()
# CP2o2=I.C.copy()
#
# I.initialize_state()
# for t in range(T):
#	I.update_P2C_o2()
# mP2Co2=I.m.copy()
# CP2Co2=I.C.copy()
#
# I.initialize_state()
# for t in range(T):
#	I.update_P0_o2()
# mP0o2=I.m.copy()
# CP0o2=I.C.copy()
#
# I.initialize_state()
# for t in range(T):
# I.update_P4_o1()
# mP4o1=I.m.copy()
# CP4o1=I.C.copy()
#
# I.initialize_state()
# for t in range(T):
# I.update_P4_o2()
# mP4o2=I.m.copy()
# CP4o2=I.C.copy()
##
# I.initialize_state()
# for t in range(T):
# I.update_P5_o2()
# mP5o2=I.m.copy()
# CP5o2=I.C.copy()
#
# plt.figure()
# plt.plot(sorted(m_exp),sorted(m_exp),'k.-',lw=1)
# plt.plot(m_exp,mP0o2,'ro',ms=10)
# plt.plot(m_exp,mP2o2,'b*')
# plt.plot(m_exp,mP2Co2,'r.')
# plt.plot(m_exp,mP4o1,'go',ms=8)
# plt.plot(m_exp,mP4o2,'r*')
# plt.plot(m_exp,mP5o2,'g.')
# plt.figure()
# plt.plot(C_exp[iu1],CP0o2[iu1],'bo',ms=10)
# plt.plot(C_exp[iu1],CP2o2[iu1],'b*')
# plt.plot(C_exp[iu1],CP2Co2[iu1],'r.')
# plt.plot(C_exp[iu1],CP4o1[iu1],'go')
# plt.plot(C_exp[iu1],CP4o2[iu1],'g*')
# plt.plot(C_exp[iu1],CP5o2[iu1],'g.')
#
# plt.plot(sorted(C_exp[iu1]),sorted(C_exp[iu1]),'k.-',lw=1)
