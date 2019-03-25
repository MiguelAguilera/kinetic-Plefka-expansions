#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 17:08:40 2018

@author: Miguel Aguilera
"""

import numpy as np
from ising_functions import *


class mf_ising:
    """
    This class implements the behaviour of a mean field approximation of an
    asymmetric kinetic Ising model.
    It can use either the naive mean field (nMF) or the Thouless-Anderson-Palmer
    equations (TAP) equations to compute the evolution of the means and correlations
    of the model.
    It also implements a function to compute an approximation of the integrated
    information of the system using the mean field approximations to compute the
    Fisher information of the system as an approximation of the Kullback Leibler
    divergence between the system and its minimum information partition (MIP)
    """

    def __init__(self, netsize, mode='nMF', update_rule=3, order=1):  # Create ising model

        self.size = netsize
        self.H = np.zeros(netsize)					# Local fields
        # Couplings							# Inverse temperature
        self.J = np.zeros((netsize, netsize))
        # Derivatives of m respect J to compute Fisher information
        self.initialize_state()
        self.coupled = False

        if mode == 'nMF':							# Naive mean field equations
            self.mode = 'nMF'
        elif mode == 'TAP':						# Thouless-Anderson-Palmer equations
            self.mode = 'TAP'
        else:
            print("Error. Mode should be either 'nMF' or 'TAP'.")
            exit(1)

    def randomize_initial_state(self):
        self.m = np.random.randint(0, 2, self.size) * 2 - 1

    def initialize_state(self, m=None, C=None):
        if m is None:
            self.m = np.zeros(self.size)
        else:
            self.m = m
        self.m_p = self.m.copy()
        if C is None:
            self.C = np.zeros((self.size, self.size)) + np.diag(1 - self.m**2)
        else:
            self.C = C
        self.C_p = self.C.copy()
        self.D = np.zeros((self.size, self.size))

    def set_initial_derivatives(self):
        self.dm = np.zeros((self.size, self.size, self.size))
        self.dC = np.zeros((self.size, self.size, self.size, self.size))

    def random_fields(self, amp=1):
        """
        Set random values for H
        """
        self.H = (np.random.rand(self.size) * 2 - 1) * amp

    def random_wiring(self, std=1):
        """
        Set random values for J
        """
        self.J = np.random.randn(self.size, self.size) * std / self.size

    def update_P0_o1(self):
        """
        Update mean field P[t-1:t] order 1 approximation
        """
        self.m_p = self.m.copy()
        self.m = update_m_P0_o1(self.H, self.J, self.m_p)
        self.C = update_C_P0_o1(self.H, self.J, self.m)

    def update_P0_o2(self):
        """
        Update mean field P[t-1:t] order 2 approximation
        """
        self.m_p = self.m.copy()
        self.m = update_m_P0_o2(self.H, self.J, self.m_p)
        self.C = update_C_P0_o2(self.H, self.J, self.m, self.m_p)


    def update_P0_o2_(self):
        """
        Update mean field P[t-1:t] order 2 approximation without neglecting order 3 terms
        """
        self.m_p = self.m.copy()
        self.m = update_m_P0_o2_(self.H, self.J, self.m_p)
        self.C = update_C_P0_o2_(self.H, self.J, self.m, self.m_p)

    def update_P1_o1(self):
        """
        Update mean field P[t] order 1 approximation
        """
        self.m_p = self.m.copy()
        self.m = update_m_P1_o1(self.H, self.J, self.m_p)
        self.C = update_C_P1_o1(self.H, self.J, self.m)

    def update_P1_o2(self):
        """
        Update mean field P[t] order 2 approximation
        """
        self.m_p = self.m.copy()
        self.C_p = self.C.copy()
#		self.m = update_m_P2_o1(self.H, self.J, self.m_p)
        self.m = update_m_P1_o2(self.H, self.J, self.m_p, self.C_p)
        self.C = update_C_P1_o2(self.H, self.J, self.m, self.C_p)

    def update_P1_o2_(self):
        """
        Update mean field P[t] order 2 approximation without neglecting order 3 terms
        """
        self.m_p = self.m.copy()
        self.C_p = self.C.copy()
#		self.m = update_m_P2_o1(self.H, self.J, self.m_p)
        self.m = update_m_P1_o2_(self.H, self.J, self.m_p, self.C_p)
        self.C = update_C_P1_o2_(self.H, self.J, self.m, self.C_p)

    def update_P1C_o2(self):
        """
        Update pairwise P[t] order 2 approximation
        """
        self.m_p = self.m.copy()
        self.C_p = self.C.copy()
        V_p = np.einsum(
            'ij,kl,jl->ik',
            self.J,
            self.J,
            self.C_p,
            optimize=True)
#
        self.m, self.C = update_P1C_o2(self.H, self.J, self.m, self.m_p, V_p)
#		self.m = update_m_P2_o2(self.H, self.J, self.m_p, self.C_p)
#		print()
#		print(self.m_p)
#		print((self.C_p))
#		print((V_p))
#	def update_P02_o2(self):
#		self.m_p = self.m.copy()
#		self.C_p = self.C.copy()
#		self.m = update_m_P0_o2(self.H, self.J, self.m_p)
#		self.C = update_C_P2_o2(self.H, self.J, self.m,self.C_p)

    def update_P2_o1(self):
        """
        Update mean field P[t-1] order 1 approximation
        """
        self.m_p = self.m.copy()
        self.m = update_m_P2_o1(self.H, self.J, self.m_p)
        self.C = update_C_P2_o1(self.H, self.J, self.m, self.m_p)
#
#    def update_P2_o2(self):
#        self.m_pp = self.m_p.copy()
#        self.m_p = self.m.copy()
#        self.C_pp = self.C_p.copy()
#        self.C_p = self.C.copy()
#        V_pp1 = np.einsum(
#            'ij,kj,j->ik',
#            self.J,
#            self.J,
#            1 - self.m_pp**2,
#            optimize=True)
#        self.m = update_m_P2_o2(self.H, self.J, self.m_p, V_pp1)
#        self.C = update_C_P2_o2(self.H, self.J, self.m, self.m_p, V_pp1)

    def update_P2_o2(self):
        """
        Update mean field P[t-1] order 2 approximation
        """
        self.m_pp = self.m_p.copy()
        self.m_p = self.m.copy()
        self.C_pp = self.C_p.copy()
        self.C_p = self.C.copy()
        V_pp = np.einsum(
            'ij,kl,jl->ik',
            self.J,
            self.J,
            self.C_pp,
            optimize=True)
        self.m = update_m_P2_o2(self.H, self.J, self.m_p, V_pp)
        self.C = update_C_P2_o2(self.H, self.J, self.m, self.m_p, V_pp)

    def update_nMF(self):
        """
        Update the mean magnetizations of the system according to the nMF equations
        m_i(t) = \tanh[β(H_i(t-1)+\sum_j J_{ij}m_j(t-1))]
        """
        self.m_p = self.m.copy()
        self.m = np.tanh((self.H + np.dot(self.J, self.m)))

    def update_means(self, Tmax=1E2):
        """
        Update the mean magnetizations of the system according to the TAP equations
        m_i(t) = \tanh[ β  (H_i(t-1)+\sum_j J_{ij} m_j(t-1))
        - m_i(t) β^2 (\sum_j J_{ij}^2(1-m_j(t-1)^2))
        """
#		m_new = self.m.copy()
#		m_new = np.random.rand(self.size) * 2 - 1
#		error = 1
#		t=0
        if self.coupled:
            sC = np.einsum(
                'ij,il,jl->i',
                self.J,
                self.J,
                self.C,
                optimize=True)
        else:
            sC = np.einsum('ij,j->i', self.J**2, 1 - self.m**2, optimize=True)
#		sC = 0.9*np.einsum('ij,il,jl->i',self.J,self.J,self.C, optimize=True) \
#			+0.1*np.einsum('ij,j->i',self.J**2,1-self.m**2, optimize=True)
#		print('sC',sC)
        H = self.H + np.dot(self.J, self.m)
        self.m_p = self.m.copy()
        self.m = scipy.optimize.fsolve(TAP_eq, self.m.copy(), args=(H, sC))
#		self.m = np.clip(-1/2/np.sqrt(sC),1/2/np.sqrt(sC),self.m)
#		for i in range(self.size):
#
#			x = scipy.optimize.fsolve(TAP_eq, self.m, args=(H,sC), xtol=1e-14)
#			print(x)
#		print('-----')
#		while error > 1E-10:
#			m_new1 = m_new.copy()
# m_new = np.tanh((self.H + np.dot(self.J, self.m)) -
# m_new * (np.dot(self.J**2, 1 - self.m**2)))
#			self.Delta_H = m_new * sC
# print(self.Delta_H)
#			m_new = np.tanh((self.H + np.dot(self.J, self.m)) - self.Delta_H)
#			m_new = np.clip(m_new,-1,1)
#			print(m_new)
#			error = np.mean(np.sqrt((m_new - m_new1)**2))
#			t+=1
#			if t>Tmax:
#				print('Warning. TAP equation did not converge.')
#				break
##				m_new = np.random.rand(self.size) * 2 - 1
# t=0
#		self.m_p = self.m.copy()
#		self.m = m_new.copy()

#	def update_means(self):
#		"""
#		Update the mean magnetizations of the system
#		"""
#		if self.mode == 'nMF':
#			self.update_nMF()
#		elif self.mode == 'TAP':
#			self.update_TAP()

#	def get_A(self):
#		"""
#		Get the A matrix for updating the correlations of the system
#		"""
#		if self.mode == 'nMF':
#			a = (1 - self.m**2)
#		elif self.mode == 'TAP':
#			a = (1 - self.m**2) * (1 - self.beta**2
#				   * (1 - self.m**2) * np.dot(self.J**2, 1 - self.m_p**2))
#		self.A = np.diag(a)

    def update_correlations(self):
        """
        Update the covariance matrix of the system.
        It should be executed after update_means
        """
#		self.get_A()
        self.C_p = self.C.copy()

        self.A = np.diag((1 - self.m**2))
        self.C_p = self.C.copy()
        self.C = np.einsum(
            'i,k,ij,kl,jl->ik',
            1 - self.m**2,
            1 - self.m**2,
            self.J,
            self.J,
            self.C_p,
            optimize=True)
#		self.C = np.einsum('i,k,ij,kj,j->ik',1 - self.m**2, 1 - self.m**2, self.J, self.J, 1-self.m_p**2, optimize=True)
#		self.C =(0.9*np.einsum('i,k,ij,kl,jl->ik',1 - self.m**2, 1 - self.m**2, self.J, self.J, self.C_p, optimize=True) \
#			+ 0.1*np.einsum('i,k,ij,kj,j->ik',1 - self.m**2, 1 - self.m**2, self.J, self.J, 1-self.m_p**2, optimize=True) )
        np.einsum('ii->i', self.C, optimize=True)[:] = 1 - self.m**2
        maxC = 1 - np.abs(np.einsum('i,j->ij', self.m, self.m))
        minC = -1 + np.abs(np.einsum('i,j->ij', self.m, self.m))
        self.C = np.clip(self.C, minC, maxC)

    def update_delayed_correlations(self):
        """
        Update the covariance matrix of the system.
        It should be executed after update_means
        """
#		self.get_A()
        self.D_p = self.D.copy()
        self.D = np.einsum(
            'i,ij,jl->il',
            1 - self.m**2,
            self.J,
            self.C_p,
            optimize=True)
#
#		maxC = 1 - np.abs(np.einsum('i,j->ij',self.m,self.m))
#		minC = -1 + np.abs(np.einsum('i,j->ij',self.m,self.m))
#		self.C = np.clip(self.C,minC,maxC)

    def update_derivatives(self):
        """
        Update the derivatives of the mean magnetizations of the system
        d m_k / d J_ij necessary for computing the Fisher information.
        It should be executed after update_means
        """
        self.dm_p = self.dm.copy()
        self.dC_p = self.dC.copy()
        self.dm = np.zeros((self.size, self.size, self.size))
        self.dC = np.zeros((self.size, self.size, self.size, self.size))

        np.einsum('kkj->kj',
                  self.dm,
                  optimize=True)[:] += np.einsum('k,j->kj',
                                                 (1 - self.m**2) * (1 - 2 * self.m * self.Delta_H),
                                                 self.m_p,
                                                 optimize=True)
        np.einsum('kkj->kj', self.dm, optimize=True)[:] += np.einsum(
            'k,kl,jl->kj', -(1 - self.m**2) * 2 * self.m, self.J, self.C_p, optimize=True)
        self.dm += np.einsum('k,kl,lij->kij', (1 - self.m**2),
                             self.J, self.dm_p, optimize=True)
        self.dm += np.einsum('k,kl,kn,lnij->kij',
                             (1 - self.m**2) * self.m,
                             self.J,
                             self.J,
                             self.dC_p,
                             optimize=True)

        np.einsum('kkj->kj',
                  self.dm,
                  optimize=True)[:] += np.einsum('k,j->kj',
                                                 (1 - self.m**2) * (1 - 2 * self.m * self.Delta_H),
                                                 self.m_p,
                                                 optimize=True)
        np.einsum('kkj->kj', self.dm, optimize=True)[:] += np.einsum(
            'k,kl,jl->kj', -(1 - self.m**2) * 2 * self.m, self.J, self.C_p, optimize=True)

        np.einsum('kmkj->kmj',
                  self.dC,
                  optimize=True)[:] += np.einsum('k,m,kl,jl->kmj',
                                                 (1 - self.m**2),
                                                 (1 - self.m**2),
                                                 self.J,
                                                 self.C_p,
                                                 optimize=True)
        np.einsum('kmmj->kmj',
                  self.dC,
                  optimize=True)[:] += np.einsum('k,m,ml,jl->kmj',
                                                 (1 - self.m**2),
                                                 (1 - self.m**2),
                                                 self.J,
                                                 self.C_p,
                                                 optimize=True)
        np.einsum('kkkj->kj',
                  self.dC,
                  optimize=True)[:] = 2 * np.einsum('i,j->ij',
                                                    (1 - self.m**2)**2 * self.Delta_H,
                                                    self.m_p,
                                                    optimize=True)
        self.dC += np.einsum('k,m,kl,mn,lnij->kmij',
                             (1 - self.m**2),
                             (1 - self.m**2),
                             self.J,
                             self.J,
                             self.dC_p,
                             optimize=True)
        self.dC += 0.5 * np.einsum('k,ml,lij->kmij',
                                   (1 - self.m**2),
                                   (1 - self.m**2),
                                   self.Delta_H,
                                   self.J,
                                   self.dm_p,
                                   optimize=True)
        self.dC += 0.5 * np.einsum('m,kl,lij->kmij',
                                   (1 - self.m**2),
                                   (1 - self.m**2),
                                   self.Delta_H,
                                   self.J,
                                   self.dm_p,
                                   optimize=True)


#		if self.mode == 'nMF':
#			"""
#			\frac{d m(t)_k}{d J_{ij}} =
#			(1-m^{[\tau]~2}_k) β (m(t-1)_j + Δ^k_{ij}[m(t-1)]), if i==k
#			(1-m^{[\tau]~2}_k) β Δ^k_{ij}[m(t-1)], otherwise
#			"""
#			np.einsum('kkj->kj', self.dm, optimize=True)[:] += np.einsum('k,j->kj',(1 - self.m**2), self.m_p, optimize=True)
#			Delta_k_ij = np.einsum('kl,lij->kij',self.J, self.dm_p, optimize=True)
#			self.dm += np.einsum('k,kij->kij',(1 - self.m**2), Delta_k_ij, optimize=True)
##			self.dm += np.einsum('k,kl,lij->kij',(1 - self.m**2),self.J, self.dm_p, optimize=True)
#		if self.mode == 'TAP':
#			"""
#			\frac{d m(t)_k}{d J_{ij}} =
#			(1-m(t)^2_k) β (m(t-1)_j + Δ^k_{ij}[m(t-1)]
#			- β m(t)_k (2 J_{ij} (1-m^{[\tau-1]~2}_j) ) , if i==k
#			(1-m(t)^2_k) β Δ^k_{ij}[m(t-1)], otherwise
#			"""
#			G_k = 1 / (1 + np.einsum('k,kl,l->k',\
#				(1 - self.m**2),self.J**2,1 - self.m_p**2, optimize=True) )
#			np.einsum('kkj->kj', self.dm, optimize=True)[:] += np.einsum('k,j->kj',\
#			(1 - self.m**2) * G_k, self.m_p, optimize=True)
#			np.einsum('kkj->kj', self.dm, optimize=True)[:] -= np.einsum('k,kj,j->kj', \
#				(1 - self.m**2)**2 * G_k * self.m , \
#				2 * self.J, 1 - self.m_p**2, optimize=True)
#			Delta_k_ij = np.einsum('kl,lij->kij',self.J, self.dm_p, optimize=True)
#			Delta_k_ij += np.einsum('k,kl,l,lij->kij',2 * self.m\
#						   ,self.J**2, self.m_p, self.dm_p, optimize=True)
#			self.dm += np.einsum('k,kij->kij',(1 - self.m**2)  * G_k, Delta_k_ij, optimize=True)


#	def update_mean_derivatives_test_TAP(self,k,i,j):
#		""" \frac{d m(t)_k}{d J_{ij}} =
#		(1-m^{[\tau]~2}_k) β (m(t-1)_j + Δ^{k~TAP}_{ij}[m(t-1)]
#		- β m(t)_k (2 J_{ij} (1-m^{[\tau-1]~2}_j) ) , if i==k
#		(1-m^{[\tau]~2}_k) β Δ^{k~TAP}_{ij}[m(t-1)], otherwise
#			"""
#		dm=0
#		G_k = 1 / (1 + self.beta**2  * (1 - self.m[k]**2) * np.sum(self.J[k,:]**2 * (1 - self.m_p**2)))
#		if j==i:
#			dm = (1 - self.m[k]**2) * G_k * (self.m_p[j] \
#					- self.m[k] * 2 * self.J[k,j] * (1 - self.m_p[j]**2))
#		Delta_k_ij = np.dot(self.J[k,:], self.dm_p[:,i,j]) \
#			+ 2 * self.m[k] * np.sum(self.J[k,:]**2 * (1 - self.m_p) * self.dm_p[:,i,j])
#		dm += (1 - self.m[k]**2) * G_k * Delta_k_ij
#		return dm
#		for k in range(self.size):
#			if self.mode == 'nMF':
#				""" \frac{d m(t)_k}{d J_{ij}} =
#				(1-m^{[\tau]~2}_k) β (m(t-1)_j + Δ^{k~nMF}_{ij}[m(t-1)]), if i==k
#				(1-m^{[\tau]~2}_k) β Δ^{k~nMF}_{ij}[m(t-1)], otherwise
#				"""
#
#				Delta_k_ij1 = np.einsum('l,lij',self.J[k, :],self.dm_p)
#				error =  np.max(np.abs(Delta_k_ij1 - Delta_k_ij[k,:,:]))
#				if error>0:
#					print('error',error)

#				self.dm[k, :, :] += (1 - self.m[k]**2) * Delta_k_ij[k,:,:]
#			if self.mode == 'TAP':
#				""" \frac{d m(t)_k}{d J_{ij}} =
#				(1-m^{[\tau]~2}_k) β (m(t-1)_j + Δ^{k~TAP}_{ij}[m(t-1)]
#				- β m(t)_k (2 J_{ij} (1-m^{[\tau-1]~2}_j) ) , if i==k
#				(1-m^{[\tau]~2}_k) β Δ^{k~TAP}_{ij}[m(t-1)], otherwise
#				"""
#				self.dm[k, k, :] += (1 - self.m[k]**2) * (self.m_p \
#					- self.m[k] * 2 * self.J[k,:] * (1 - self.m_p**2))
#				Delta_k_ij = np.dot(self.J[k,:], self.dm_p) \
#					- self.dm_p[k,:,:] * np.dot(self.J[k,:]**2,1 - self.m_p**2) \
#					+ 2 * self.m[k] * np.dot(self.J[k,:]**2 * (1 - self.m_p), self.dm_p)
#				self.dm[k, :, :] += (1 - self.m[k]**2) * Delta_k_ij

#			if self.mode == 'nMF':
#				error =  np.max(np.abs(Delta_k_ij2 - Delta_k_ij))
#				if error>0:
#					print('error',error)
#					plt.figure()
#					plt.imshow(Delta_k_ij2[:,:])
#					plt.figure()
#					plt.imshow(Delta_k_ij)
#					plt.figure()
#					plt.plot(Delta_k_ij2[0,:])
#					plt.plot(Delta_k_ij[0,:])

#		dm_k = np.tile((1 - self.m**2),(self.size,self.size,1)).T
#		mj_i_eq_k = np.tile(np.diag(self.m_p),(self.size,1,1)).T
#		Delta_k_ij = np.tile(selfJ,(self.size,1,1)) * self.dm
#		self.dm = * ()
#		 np.tile(np.expand_dims(np.expand_dims(a,1),2),(1,3,3))


    def find_stable_point(self, max_error=1E-10, Tmax=1E4, compute_dm=False):
        error = 1
        t = 0
        while error > max_error and t < Tmax:
            self.update_means()
            self.update_correlations()

            t += 1
            error_m = np.mean(np.sqrt((self.m - self.m_p)**2))
            error_C = np.mean(np.sqrt((self.C - self.C_p)**2))
            error = max(error_m, error_C)
            if compute_dm:
                self.update_mean_derivatives()
                error_dm = np.mean(np.sqrt((self.dm - self.dm_p)**2))
                error = max(error, error_dm)

    def run(self, m0, T):
        """
        Run the model for T steps from a initial state m0 and compute means,
        correlations and derivative of means
        """
        self.m = m0.copy()
        self.C = np.zeros((self.size, self.size)) + np.diag(1 - self.m**2)
        self.dm = np.zeros((self.size, self.size, self.size))
        for i in range(T):
            self.update_correlations()
            self.update_mean_derivatives()


#	def Fisher_information(self, i, j, k, l):
#		"""
#		F[J_{ij},J_{kl}] = β^2 (<s(t)_i s(t)_k> - <s(t)_i> <s(t)_k> )
#		* (m(t-1)_j + \sum_u  J_{iu} * {dm(t-1)_u}/{dJ_{ij}} )
#		* (m(t-1)_l + \sum_v  J_{kv} * {dm(t-1)_v}/{dJ_{kl}} )
#		"""
#		if np.isscalar(i) and np.isscalar(j) and np.isscalar(k) and np.isscalar(l):
#			F = self.beta** 2 * self.C[i,k]*(self.m_p[j] + np.dot(self.J[i,:],self.dm_p[:,i,j])) \
#					* (self.m_p[l] + np.dot(self.J[k,:],self.dm_p[:,k,l]))
#		else:
#			F = self.C[i,k]*(self.m_p[j] + np.einsum('xy,yx->x',self.J[i,:],self.dm_p[:,i,j], optimize = True)) \
#					* (self.m_p[l] + np.einsum('xy,yx->x',self.J[k,:],self.dm_p[:,k,l], optimize = True))
#		return(F)

    def Fisher_matrix(self):
        """
        F[J_{ij},J_{kl}] = β^2 (<s(t)_i s(t)_k> - <s(t)_i> <s(t)_k> )
        * (m(t-1)_j + \sum_u  J_{iu} * {dm(t-1)_u}/{dJ_{ij}} )
        * (m(t-1)_l + \sum_v  J_{kv} * {dm(t-1)_v}/{dJ_{kl}} )
        """

        G = np.einsum('mn,nij->mij', self.J, self.dm_p)

        F = np.einsum('m,mij, mkl,-> ijkl', 1 - self.m**2, G, G)
        F += np.einsum('i,j,iij->ijkl', 1 - self.m**2, m_p, G)
        F += np.einsum('k,l,kkl->ijkl', 1 - self.m**2, m_p, G)
        np.einsum('ijil->ijl',
                  F,
                  optimize=True)[:] += 2 * np.einsum('i,j,l->ijl',
                                                     2 * (1 - self.m**2)**2,
                                                     self.m_p,
                                                     self.m_p,
                                                     optimize=True)

        return F
#
#
#		G = np.einsum('ik,j,l->ijkl',self.C, self.m_p, self.m_p, optimize = True)
#		G += np.einsum('ik,j,kv,vkl->ijkl',self.C, self.m_p, self.J,self.dm_p, optimize = True)
#		G += np.einsum('ik,iu,uij,l->ijkl',self.C, self.J, self.dm_p, self.m_p, optimize = True)
#		G += np.einsum('ik,iu,uij,kv,vkl->ijkl',self.C, self.J, self.dm_p, self.J,self.dm_p, optimize = True)
#		return(G)
#

    def Fisher_terms(self):
        """
        F[J_{ij},J_{kl}] = β^2 (<s(t)_i s(t)_k> - <s(t)_i> <s(t)_k> )
        * (m(t-1)_j + \sum_u  J_{iu} * {dm(t-1)_u}/{dJ_{ij}} )
        * (m(t-1)_l + \sum_v  J_{kv} * {dm(t-1)_v}/{dJ_{kl}} )
        """
        G = 0.5**2 * np.einsum('ik,j,l,ij,kl->ijkl',
                               self.C,
                               self.m_p,
                               self.m_p,
                               self.J,
                               self.J,
                               optimize=True)
        G += 0.5**2 * np.einsum('ik,j,kv,vkl,ij,kl->ijkl',
                                self.C,
                                self.m_p,
                                self.J,
                                self.dm_p,
                                self.J,
                                self.J,
                                optimize=True)
        G += 0.5**2 * np.einsum('ik,iu,uij,l,ij,kl->ijkl',
                                self.C,
                                self.J,
                                self.dm_p,
                                self.m_p,
                                self.J,
                                self.J,
                                optimize=True)
        G += 0.5**2 * np.einsum('ik,iu,uij,kv,vkl,ij,kl->ijkl',
                                self.C,
                                self.J,
                                self.dm_p,
                                self.J,
                                self.dm_p,
                                self.J,
                                self.J,
                                optimize=True)
        return(G)


#	def Fisher_terms(self):
#		"""
#		F[J_{ij},J_{kl}] = β^2 (<s(t)_i s(t)_k> - <s(t)_i> <s(t)_k> )
#		* (m(t-1)_j + \sum_u  J_{iu} * {dm(t-1)_u}/{dJ_{ij}} )
#		* (m(t-1)_l + \sum_v  J_{kv} * {dm(t-1)_v}/{dJ_{kl}} )
#		"""
#		G = 0.5**2 * np.einsum('ik,j,l,ij,kl->ijkl',self.C, self.m_p, self.m_p, np.ones((self.size,self.size)), np.ones((self.size,self.size)), optimize = True)
#		G += 0.5**2 * np.einsum('ik,j,kv,vkl,ij,kl->ijkl',self.C, self.m_p, self.J,self.dm_p, np.ones((self.size,self.size)), np.ones((self.size,self.size)), optimize = True)
#		G += 0.5**2 * np.einsum('ik,iu,uij,l,ij,kl->ijkl',self.C, self.J, self.dm_p, self.m_p, np.ones((self.size,self.size)), np.ones((self.size,self.size)), optimize = True)
#		G += 0.5**2 * np.einsum('ik,iu,uij,kv,vkl,ij,kl->ijkl',self.C, self.J, self.dm_p, self.J,self.dm_p, np.ones((self.size,self.size)), np.ones((self.size,self.size)), optimize = True)
#		return(G)
#


    def get_partition_inds(self, Bc, Bf):
        """
        Transform a bipartition to a list of pairs of indices of the couplings
        affected by the partition

        :params numpy.ndarray Bc, Bf:
        Description of a bipartion in terms of a pair of lists of nodes of one
        of the halves for the current state (Bc) and future state (Bf)
        """
        inds_i = []
        inds_j = []
        for i in range(self.size):
            for j in range(self.size):
                if (i in Bf) ^ (j in Bc):
                    inds_i += [i]
                    inds_j += [j]
        return np.array(inds_i), np.array(inds_j)

    def phi(self, Bc, Bf):
        """
        \varphi^{cut} = D[ p(s(t)|s(0)) || p^{cut}(s(t)|s(0)) ]
        ≈ \sum_{J_{ij},J_{kl} \in {J}_{cut}} F(t)[J_{ij},J_{kl}]*ΔJ_{ij}*ΔJ_{kl}

        :params numpy.ndarray Bc, Bf:
        Description of a bipartion in terms of a pair of lists of nodes of one
        of the halves for the current state (Bc) and future state (Bf)
        """
        inds_i, inds_j = self.get_partition_inds(Bc, Bf)
        inds_k = inds_i.copy()
        inds_l = inds_j.copy()

        phi = 0.5**2 * np.einsum('xy,x,y,x,y->',
                                 self.C[inds_i,
                                        :][:,
                                           inds_k],
                                 self.m_p[inds_j],
                                 self.m_p[inds_l],
                                 self.J[inds_i,
                                        inds_j],
                                 self.J[inds_k,
                                        inds_l],
                                 optimize=True)
        phi += 0.5**2 * np.einsum('xy,x,yv,vy,x,y->',
                                  self.C[inds_i,
                                         :][:,
                                            inds_k],
                                  self.m_p[inds_j],
                                  self.J[inds_k,
                                         :],
                                  self.dm_p[:,
                                            inds_k,
                                            inds_l],
                                  self.J[inds_i,
                                         inds_j],
                                  self.J[inds_k,
                                         inds_l],
                                  optimize=True)
        phi += 0.5**2 * np.einsum('xy,xu,ux,y,x,y->',
                                  self.C[inds_i,
                                         :][:,
                                            inds_k],
                                  self.J[inds_i,
                                         :],
                                  self.dm_p[:,
                                            inds_i,
                                            inds_j],
                                  self.m_p[inds_l],
                                  self.J[inds_i,
                                         inds_j],
                                  self.J[inds_k,
                                         inds_l],
                                  optimize=True)
        phi += 0.5**2 * np.einsum('xy,xu,ux,yv,vy,x,y->',
                                  self.C[inds_i,
                                         :][:,
                                            inds_k],
                                  self.J[inds_i,
                                         :],
                                  self.dm_p[:,
                                            inds_i,
                                            inds_j],
                                  self.J[inds_k,
                                         :],
                                  self.dm_p[:,
                                            inds_k,
                                            inds_l],
                                  self.J[inds_i,
                                         inds_j],
                                  self.J[inds_k,
                                         inds_l],
                                  optimize=True)
#
        return phi
#

    def search_MIP(self):

        Bc = np.array([])
        Bf = np.array([])
        min_phi = np.inf
        Bc_min = Bc.copy()
        Bf_min = Bf.copy()

        found = False
        for n in range(
                self.size):			# There is 2 * size nodes to be partitioned, but just searching half of the space is enought
            print(len(Bc), len(Bf))
            phis = []
            inds = []

            for t in range(self.size):
                if t not in Bc:
                    phis += [self.phi(np.append(Bc, t), Bf)]
                    inds += [(0, t)]
                if t not in Bf:
                    phis += [self.phi(Bc, np.append(Bf, t))]
                    inds += [(1, t)]
            print(phis)
            print(inds)

            min_ind = np.argmin(phis)
            print(min_ind)
            print(inds[min_ind])

            if inds[min_ind][0] == 0:
                print('Bc')
                Bc = np.append(Bc, inds[min_ind][1])
            elif inds[min_ind][0] == 1:
                print('Bf')
                Bf = np.append(Bf, inds[min_ind][1])

            if phis[min_ind] < min_phi:
                print('found')

                min_phi = phis[min_ind]
                Bc_min = Bc.copy()
                Bf_min = Bf.copy()

            print(Bc, Bf, phis[min_ind])
            print(Bc_min, Bf_min, min_phi)


class mf_ising_roudi:
    """
    This class implements the behaviour of a mean field approximation of an
    asymmetric kinetic Ising model.
    It can use either the naive mean field (nMF) or the Thouless-Anderson-Palmer
    equations (TAP) equations to compute the evolution of the means and correlations
    of the model.
    It also implements a function to compute an approximation of the integrated
    information of the system using the mean field approximations to compute the
    Fisher information of the system as an approximation of the Kullback Leibler
    divergence between the system and its minimum information partition (MIP)
    """

    def __init__(self, netsize, mode='TAP'):  # Create ising model

        self.size = netsize
        self.H = np.zeros(netsize)					# Local fields
        # Couplings							# Inverse temperature
        self.J = np.zeros((netsize, netsize))
        # Derivatives of m respect J to compute Fisher information
        self.initialize_state()

        if mode == 'nMF':							# Naive mean field equations
            self.mode = 'nMF'
        elif mode == 'TAP':						# Thouless-Anderson-Palmer equations
            self.mode = 'TAP'
        else:
            print("Error. Mode should be either 'nMF' or 'TAP'.")
            exit(1)

    def randomize_initial_state(self):
        self.m = np.random.randint(0, 2, self.size) * 2 - 1

    def initialize_state(self, m=None, C=None):
        if m is None:
            self.m = np.zeros(self.size)
            self.C = np.zeros((self.size, self.size)) + np.diag(1 - self.m**2)
        elif C is None:
            self.m = m
            self.C = np.zeros((self.size, self.size)) + np.diag(1 - self.m**2)
        else:
            self.m = m
            self.C = C
        self.D = np.zeros((self.size, self.size)) + np.diag(1 - self.m**2)

    def random_fields(self, amp=1):
        """
        Set random values for H
        """
        self.H = (np.random.rand(self.size) * 2 - 1) * amp

    def random_wiring(self, std=1):
        """
        Set random values for J
        """
        self.J = np.random.randn(self.size, self.size) * std / self.size

    def update_nMF(self):
        """
        Update the mean magnetizations of the system according to the nMF equations
        m_i(t) = \tanh[β(H_i(t-1)+\sum_j J_{ij}m_j(t-1))]
        """
        self.m_p = self.m.copy()
        self.m = np.tanh((self.H + np.dot(self.J, self.m)))

    def update_TAP(self, Tmax=1E2):
        """
        Update the mean magnetizations of the system according to the TAP equations
        m_i(t) = \tanh[ β  (H_i(t-1)+\sum_j J_{ij} m_j(t-1))
        - m_i(t) β^2 (\sum_j J_{ij}^2(1-m_j(t-1)^2))
        """
        sC = np.einsum('ij,j->i', self.J**2, 1 - self.m**2, optimize=True)
#		print('sC',sC)
        H = self.H + np.dot(self.J, self.m)
        self.m_p = self.m.copy()
        self.m = scipy.optimize.fsolve(TAP_eq, self.m, args=(H, sC))
#		m_new = self.m.copy()
#		m_new = np.random.rand(self.size) * 2 - 1
#		error = 1
#		t=0
#		while error > 1E-10:
#			m_new1 = m_new.copy()
# m_new = np.tanh((self.H + np.dot(self.J, self.m)) -
# m_new * (np.dot(self.J**2, 1 - self.m**2)))
#			self.Delta_H = m_new * np.einsum('ij,j->i',self.J**2,1-self.m**2, optimize=True)
#			m_new = np.tanh((self.H + np.dot(self.J, self.m)) - self.Delta_H)
#			m_new = np.clip(m_new,-1,1)
#			error = np.mean(np.sqrt((m_new - m_new1)**2))
#			t+=1
#			if t>Tmax:
#				print('Warning. TAP equation did not converge.')
#				break
##				m_new = np.random.rand(self.size) * 2 - 1
# t=0
#		self.m_p = self.m.copy()
#		self.m = m_new.copy()

    def update_means(self):
        """
        Update the mean magnetizations of the system
        """
        if self.mode == 'nMF':
            self.update_nMF()
        elif self.mode == 'TAP':
            self.update_TAP()

    def update_correlations(self):
        """
        Update the covariance matrix of the system.
        It should be executed after update_means
        """

        a = (1 - self.m**2) * (1 - (1 - self.m**2) * \
             np.einsum('ij,j->i', self.J**2, 1 - self.m_p, optimize=True))
        self.C_p = self.C.copy()
        self.C = self.C = np.einsum(
            'i,k,ij,kl,jl->ik',
            a,
            a,
            self.J,
            self.J,
            self.C_p,
            optimize=True)
#		self.C -= np.einsum('i,il,l,k,ij,kj,j->ik',(1 - self.m**2)**2, self.J**2, 1-self.m_p**2, 1 - self.m**2, self.J, self.J, 1-self.m_p**2, optimize=True)
#		self.C -= np.einsum('i,k,kl,l,ij,kj,j->ik', 1-self.m_p**2, (1 - self.m**2)**2, self.J**2,  1 - self.m**2, self.J, self.J, 1-self.m_p**2, optimize=True)
#		self.C += np.einsum('i,il,l,k,kn,n,ij,kj,j->ik',(1 - self.m**2)**2, self.J**2, 1-self.m_p**2,(1 - self.m**2)**2, self.J**2, 1-self.m_p**2, self.J, self.J, 1-self.m_p**2, optimize=True)
        np.einsum('ii->i', self.C, optimize=True)[:] = 1 - self.m**2

    def update_delayed_correlations(self):
        """
        Update the covariance matrix of the system.
        It should be executed after update_means
        """
        self.D_p = self.D.copy()
        a = (1 - self.m**2) * (1 - (1 - self.m**2) * \
             np.einsum('ij,j->i', self.J**2, 1 - self.m_p, optimize=True))
        self.D = np.einsum('i,ij,jl->il', a, self.J, self.C_p, optimize=True)


#		self.C_p = self.C.copy()
#
#		self.A = np.diag((1 - self.m**2))
#		self.C_p = self.C.copy()
#		self.C = np.dot(
#                    np.dot(
#                        self.A,
#                        np.dot(
#                            np.dot(self.J, self.C_p),
#                            self.J.T)),
#                    self.A)
##		row, col = np.diag_indices(self.size)
##		self.C[row, col] = 1 - self.m**2
#		np.einsum('ii->i', self.C, optimize=True)[:] = 1 - self.m**2


#		M1,M2 = np.meshgrid(self.m,self.m)
#		Cmax = 1 - M1 * M2
#		self.C[self.C>Cmax]=Cmax[self.C>Cmax]
#

# N=6
#I = mf_ising(N)
#J0 = np.ones((N, N)) / N * 1.0 +  np.random.randn(N, N) / N * 0.0
# J0[0:N//2,N//2:]*=0.0
# J0[N//2:,0:N//2]*=0.0
##J0 = np.eye(N)
#
#I.J = J0.copy()
#s0 = (np.random.rand(N)<0.8).astype(int)*2-1
##s0 = np.ones(N)*0.8
# print(s0)
#I.m = s0
# T=10
# for t in range(T):
#	print(t)
#	I.update_means()
#	I.update_correlations()
#	I.update_mean_derivatives()
# I.search_MIP()


#from matplotlib import pyplot as plt
#
#N = 64
##I = mf_ising(N, 'nMF')
#I = mf_ising(N, 'TAP')
#J0 = np.ones((N, N)) / N * 1.1 +  np.random.randn(N, N) / N * 0.0
#I.J = J0.copy()
#s0 = (np.random.rand(N)<0.8).astype(int)
# print(s0)
#I.m = s0
#T0 = 100
# for t in range(T0):
#	print(t)
#	I.update_means()
#
# i=1
# j=2
# k=0
# l=1
# T=32
# for t in range(T):
#	print(t)
#	I.update_means()
#	I.update_correlations()
#	I.update_mean_derivatives()
#
#F = I.Fisher_information(i,j,k,l)*J0[i,j]*J0[k,l]
# print(F)
# i=np.array([1,2])
# j=np.array([0,0])
# k=np.array([3,4])
# l=np.array([0,0])
#F = I.Fisher_information(i,j,k,l)*J0[i,j]*J0[k,l]
# print(F)
#
##phi = I.phi(np.arange(N,dtype=int),np.zeros(N,int))
#
# i=1
# j=0
# k=i
# l=j
#F = I.Fisher_information(i,j,k,l)*J0[i,j]*J0[k,l]
# print(0.5*F)
#phi = I.phi([1],[0])
# print(phi)
#
# i=np.array([1,2,1,2])
# j=np.array([0,0,0,0])
# k=np.array([1,1,2,2])
# l=np.array([0,0,0,0])
# print()
# for n in range(4):
#	F = I.Fisher_information(i[n],j[n],k[n],l[n])*J0[i[n],j[n]]*J0[k[n],l[n]]
#	print(0.5*F)
# print()
#F = I.Fisher_information(i,j,k,l)*J0[i,j]*J0[k,l]
# print(0.5*F)
# print(0.5*np.sum(F))
# print()
#phi = I.phi(np.array([1,2]),np.array([0,0]))
# print(phi)


#from matplotlib import pyplot as plt
#
#N = 64
##I = mf_ising(N, 'nMF')
#I = mf_ising(N, 'TAP')
#J0 = np.ones((N, N)) / N * 1.1 +  np.random.randn(N, N) / N * 0.3
# J0=np.load('couplings.npy')
#I.J = J0.copy()
##s0 = np.ones(N)
# s0[0:(N//4)]=-1
#
#s0 = (np.random.rand(N)<0.8).astype(int)
# print(s0)
#I.m = s0
#T0 = 100
# for t in range(T0):
# print(t)
# I.update_means()
#s1 = I.m
# T=32
#m = np.zeros(T)
#m1 = np.zeros(T)
#m2 = np.zeros(T)
#dm1 = np.zeros(T)
#dm2 = np.zeros(T)
# k=0
# i=1
# j=2
#
# for t in range(T):
#	print(t)
#	I.update_means()
#	m[t] = I.m[k]
#	I.update_mean_derivatives()
#	dm1[t] = I.dm[k,i,j]
#I.m = s1
#I.m_p = np.zeros(N)
#
# I.J[i,j]=0
# for t in range(T):
#	I.update_means()
##	m2[t] = I.m[0]
#	m1[t] = I.m[k]
#
# plt.figure()
# plt.plot(m)
# plt.plot(m1,'r--')
# plt.plot(m-dm1*J0[i,j],'g:')
# print(J0[i,j])
#
# plt.figure()
# plt.plot(m1-m,'r--')
# plt.plot(-dm1*J0[i,j],'g:')
#
# plt.figure()
#plt.plot((m-m1 + dm1*J0[i,j])*N)
##plt.plot((m-m2 - dm2*I.J[0,0])*N)
#
# plt.show()


#N = 2
#I = mf_ising(N, 'nMF')
#
#I.J = np.random.randint(1, 3, (N, N))
#I.J = np.array([[0, 1], [2, 3]])
##s1 = np.random.randint(0, 2, N)
##s0 = np.random.randint(0, 2, N) * 2 - 1
#s0 = np.array([1, -1])
#s1 = np.array([0.5, 0.5])
#I.m_p = s0
#I.m = s1
# print(I.J)
# print()
#print(I.m, I.m_p)
# print()
# I.Fisher_information()
# print('----')
#s0 = s1
#s1 = s1
#I.m_p = s0
#I.m = s1
# print()
#print(I.m, I.m_p)
# I.Fisher_information()

#N = 128
#I = mf_ising(N, 'nMF')
# I=mf_ising(N,'TAP')
# I.random_fields()
# I.random_wiring()
#I.J = np.ones((N, N)) / N * 1.05
# I.randomize_state()
#s0 = np.random.rand(N) * 2 - 1
#s0 = np.random.randint(0, 2, N) * 2 - 1
#I.m = s0
# print(I.C)
# for rep in range(128):
#	I.update_correlations()
#	print()
#	print(I.C)
#	if np.max(I.C > 1):
#		print('error')
#		print(np.diag(I.A))
#		print(I.m)
#		print(I.C_p)
#		print(np.dot(np.dot(I.J, I.C_p), I.J.T))
#		break
