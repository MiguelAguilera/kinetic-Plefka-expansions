#!/usr/bin/env python3
"""
GPLv3 2020 Miguel Aguilera

This code is similar to "forward-Ising-problem.py", but it is used
for solving the reconstruction problem, so it saves just the values of
correlations at the end of the simulation.
It computes the solution of the forward Ising problem with different
mean-field approximation methods using either the original network
mode='f', or the inferred network mode='r' from solving the inverse
Ising problem.
The results can be displayed running "reconstruction-Ising-problem-results.py"
"""

import context
from plefka import mf_ising
import numpy as np


def nsf(num, n=4):
    """n-Significant Figures"""
    numstr = ("{0:.%ie}" % (n - 1)).format(num)
    return float(numstr)


size = 512                 # Network size
R = 1000000                # Repetitions of the simulation
H0 = 0.5                   # Uniform distribution of fields parameter
J0 = 1.0                   # Average value of couplings
Js = 0.1                   # Standard deviation of couplings

B = 201                    # Number of values of beta
T = 2**7                  # Number of simulation time steps

iu1 = np.triu_indices(size, 1)


betas = 1 + np.linspace(-1, 1, B) * 0.3

# As we need more data than what is generated in the forward problem,
# we compute the forward problem twice, for mode='f' and mode='r' for
# generating data for the forward and reconstruction problems
modes = ['f', 'r']        # Forward and reconstruction modes

for mode in modes:
    for ib in range(len(betas)):
        beta_ref = round(betas[ib], 3)

        # Load data

        filename = 'data/inverse/inverse_100_R_' + str(R) + '.npz'
        print(beta_ref, mode)

        data = np.load(filename)
        HP_t1_t = data['HP_t1_t'] * beta_ref
        JP_t1_t = data['JP_t1_t'] * beta_ref
        HP_t = data['HP_t'] * beta_ref
        JP_t = data['JP_t'] * beta_ref
        HP_t1 = data['HP_t1'] * beta_ref
        JP_t1 = data['JP_t1'] * beta_ref
        HP2_t = data['HP2_t'] * beta_ref
        JP2_t = data['JP2_t'] * beta_ref

        J = data['J'] * beta_ref
        H = data['H'] * beta_ref
        del data

        filename1 = 'data/data-H0-' + str(H0) + '-J0-' + str(J0) + '-Js-' + str(
            Js) + '-N-' + str(size) + '-R-' + str(R) + '-beta-1.0.npz'
        data1 = np.load(filename1)
        s0 = data1['s0']
        del data1

        # Run Plefka[t-1,t], order 2
        I = mf_ising(size)
        if mode == 'f':
            I.H = H.copy()
            I.J = J.copy()
        elif mode == 'r':
            I.H = HP_t1_t.copy()
            I.J = JP_t1_t.copy()
        I.initialize_state(s0)
        for t in range(T):
            print(beta_ref, mode, 'P_t1_t_o2', str(t) + '/' + str(T),
                  nsf(np.mean(I.m)), nsf(np.mean(I.C[iu1])), nsf(np.mean(I.D)))
            I.update_P_t1_t_o2()
        mP_t1_t_final = I.m
        CP_t1_t_final = I.C
        DP_t1_t_final = I.D

        # Run Plefka[t], order 2
        I = mf_ising(size)
        if mode == 'f':
            I.H = H.copy()
            I.J = J.copy()
        elif mode == 'r':
            I.H = HP_t.copy()
            I.J = JP_t.copy()
        I.initialize_state(s0)
        for t in range(T):
            print(beta_ref, mode, 'P_t_o2', str(t) + '/' + str(T),
                  nsf(np.mean(I.m)), nsf(np.mean(I.C[iu1])), nsf(np.mean(I.D)))
            I.update_P_t_o2()
        mP_t_final = I.m
        CP_t_final = I.C
        DP_t_final = I.D

        # Run Plefka2[t], order 2
        I = mf_ising(size)
        if mode == 'f':
            I.H = H.copy()
            I.J = J.copy()
        elif mode == 'r':
            I.H = HP2_t.copy()
            I.J = JP2_t.copy()
        I.initialize_state(s0)
        for t in range(T):
            print(beta_ref, mode, 'P2_t_o2', str(t) + '/' + str(T),
                  nsf(np.mean(I.m)), nsf(np.mean(I.C[iu1])), nsf(np.mean(I.D)))
            I.update_P2_t_o2()
        mP2_t_final = I.m
        CP2_t_final = I.C
        DP2_t_final = I.D

        # Run Plefka[t-1], order 1
        I = mf_ising(size)
        if mode == 'f':
            I.H = H.copy()
            I.J = J.copy()
        elif mode == 'r':
            I.H = HP_t1.copy()
            I.J = JP_t1.copy()
        I.initialize_state(s0)
        for t in range(T):
            print(beta_ref, mode, 'P_t1_o1', str(t) + '/' + str(T),
                  nsf(np.mean(I.m)), nsf(np.mean(I.C[iu1])), nsf(np.mean(I.D)))
            I.update_P_t1_o1()
        mP_t1_final = I.m
        CP_t1_final = I.C
        DP_t1_final = I.D

        # Save results to file

        filename = 'data/reconstruction/transition_' + mode + '_' + \
            str(int(round(beta_ref * 1000))) + '_R_' + str(R) + '.npz'
        np.savez_compressed(filename,
                            mP_t1_t=mP_t1_t_final,
                            mP_t=mP_t_final,
                            mP_t1=mP_t1_final,
                            mP2_t=mP2_t_final,
                            CP_t1_t=CP_t1_t_final,
                            CP_t=CP_t_final,
                            CP_t1=CP_t1_final,
                            CP2_t=CP2_t_final,
                            DP_t1_t=DP_t1_t_final,
                            DP_t=DP_t_final,
                            DP_t1=DP_t1_final,
                            DP2_t=DP2_t_final)
