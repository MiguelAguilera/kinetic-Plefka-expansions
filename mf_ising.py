#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Miguel Aguilera
"""

import numpy as np
from ising_functions import *


class mf_ising:
    """
    This class implements the behaviour of a mean field approximations of an
    asymmetric kinetic Ising model according to different Plefka Expansions.
    It can use either the naive mean field (nMF) or the Thouless-Anderson-Palmer
    equations (TAP) equations to compute the evolution of the means and correlations
    of the model.
    """

    def __init__(self, netsize):  # Create ising model

        self.size = netsize
        self.H = np.zeros(netsize)					# Local fields
        self.J = np.zeros((netsize, netsize))		# Couplings
        self.initialize_state()

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


    def random_fields(self, amp=1):
        """
        Set random values for H
        """
        self.H = (np.random.rand(self.size) * 2 - 1) * amp

    def random_wiring(self, offset =0, std=1):
        """
        Set random values for J
        """
        self.J = (offset + np.random.randn(self.size, self.size) * std) / self.size

    def update_P0_o1(self):
        """
        Update mean field Plefka[t-1:t] order 1 approximation
        """
        self.m_p = self.m.copy()
        self.m = update_m_P0_o1(self.H, self.J, self.m_p)
        self.C = update_C_P0_o1(self.H, self.J, self.m)
        self.D = update_D_P0_o1(self.H, self.J, self.m, self.m_p)

    def update_P0_o2(self):
        """
        Update mean field Plefka[t-1:t] order 2 approximation
        """
        self.m_p = self.m.copy()
        self.m = update_m_P0_o2(self.H, self.J, self.m_p)
        self.C = update_C_P0_o2(self.H, self.J, self.m, self.m_p)
        self.D = update_D_P0_o2(self.H, self.J, self.m, self.m_p)

    def update_P1_o1(self):
        """
        Update mean field Plefka[t] order 1 approximation
        """
        self.m_p = self.m.copy()
        self.C_p = self.C.copy()
        self.m = update_m_P1_o1(self.H, self.J, self.m_p)
        self.C = update_C_P1_o1(self.H, self.J, self.m)
        self.D = update_D_P1_o1(self.H, self.J, self.m, self.C_p)

    def update_P1_o2(self):
        """
        Update mean field Plefka[t] order 2 approximation
        """
        self.m_p = self.m.copy()
        self.C_p = self.C.copy()
        self.m = update_m_P1_o2(self.H, self.J, self.m_p, self.C_p)
        self.C = update_C_P1_o2(self.H, self.J, self.m, self.C_p)
        self.D = update_D_P1_o2(self.H, self.J, self.m, self.m_p, self.C_p)

    def update_P2_o1(self):
        """
        Update mean field P[t-1] order 1 approximation
        """
        self.m_p = self.m.copy()
        self.C_p = self.C.copy()
        self.m = update_m_P2_o1(self.H, self.J, self.m_p)
        self.D = update_D_P2_o1(self.H, self.J, self.m_p, self.C_p)
        self.C = update_C_P2_o1(self.H, self.J, self.m, self.m_p, self.C_p)
        print(np.mean(self.m), np.mean(self.C), np.mean(self.D))

    def update_P1C_o2(self):
        """
        Update pairwise Plefka2[t] order 2 approximation
        """
        self.m_p = self.m.copy()
        self.C_pp = self.C_p.copy()
        self.C_p = self.C.copy()
        self.D_p = self.D.copy()
        V_p = np.einsum(
            'ij,kl,jl->ik',
            self.J,
            self.J,
            self.C_p,
            optimize=True)
        self.m, self.C, self.D = update_P1D_o2(
            self.H, self.J, self.m_p, self.C_p, self.D_p)

