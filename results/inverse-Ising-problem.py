#!/usr/bin/env python3
"""
GPLv3 2020 Miguel Aguilera

This code computes the solution of the inverse Ising problem with different
mean-field approximation methods using data generated form simulation from
"generate_data.py".
The results can be displayed running "inverse-Ising-problem-results.py"
"""

import context
from plefka import mf_ising
from plefka import update_m_P_t1_t_o2, update_D_P_t1_t_o2, update_m_P_t_o2, update_D_P_t_o2
from plefka import update_m_P_t1_o1, update_D_P_t1_o1, update_D_P2_t_o2
import numpy as np
import time
from sys import argv


def nsf(num, n=4):
    """n-Significant Figures"""
    numstr = ("{0:.%ie}" % (n - 1)).format(num)
    return float(numstr)


size = 512                 # Network size
R = 1000000                # Repetitions of the simulation
H0 = 0.5                   # Uniform distribution of fields parameter
J0 = 1.0                   # Average value of couplings
Js = 0.1                   # Standard deviation of couplings

B = 21                    # Number of values of beta
T = 2**7                  # Number of simulation time steps

max_rep = 10000            # Maximum number of steps in the gradient ascent

# Learning speed parameters
etaH = 1 / 10
etaJ = 1 / size**0.5


betas = 1 + np.linspace(-1, 1, B) * 0.3
for ib in range(B):


    beta_ref = round(betas[ib], 3)        # beta_ref / beta_c value

    filename = 'data/data-H0-' + str(H0) + '-J0-' + str(J0) + '-Js-' + str(
        Js) + '-N-' + str(size) + '-R-' + str(R) + '-beta-' + str(beta_ref) + '.npz'
    data = np.load(filename)

    # We load original parameters for reference
    H = data['H']
    J = data['J']

    # We compute moments averaged over trials and trajectories
    m_exp = data['m']
    C_exp = data['C']
    D_exp = data['D']
    mp_exp = np.zeros((size, T))
    mp_exp[:, 1:] = m_exp[:, 0:-1]
    mp_exp[:, 0] = data['s0']
    c_exp = C_exp + np.einsum('it,jt->ijt', m_exp, m_exp, optimize=True)
    d_exp = D_exp + np.einsum('it,jt->ijt', m_exp, mp_exp, optimize=True)
    c_exp[range(size), range(size)] = 1

    W = T - 3     # We exclude the first two samples with 0 covariance t=0 and t=1
    dp_mean = np.mean(d_exp[:, :, T - W - 1:T - 1], axis=2)
    cp_mean = np.mean(c_exp[:, :, T - W - 1:T - 1], axis=2)
    mp_mean = np.mean(m_exp[:, T - W - 1:T - 1], axis=1)
    mpp_mean = np.mean(mp_exp[:, T - W - 1:T - 1], axis=1)
    d_mean = np.mean(d_exp[:, :, T - W:T], axis=2)
    c_mean = np.mean(c_exp[:, :, T - W:T], axis=2)
    m_mean = np.mean(m_exp[:, T - W:T], axis=1)
    C_mean = c_mean - np.einsum('i,j->ij', m_mean, m_mean, optimize=True)
    D_mean = d_mean - np.einsum('i,j->ij', m_mean, mp_mean, optimize=True)
    Cp_mean = cp_mean - np.einsum('i,j->ij', mp_mean, mp_mean, optimize=True)
    Dp_mean = dp_mean - np.einsum('i,j->ij', mp_mean, mpp_mean, optimize=True)

    s0 = np.array(data['s0'])
    del data

    # Reference error for finishing gradient descent
    ref_error = 1E-12
    print('ref error:', ref_error)


# Run Plefka[t-1,t], order 2

    time_start = time.perf_counter()
    I = mf_ising(size)
    I.H = np.zeros(size)
    I.J = np.zeros((size, size))
    rep = 0
    cond = True
    while cond:
        I.m_p = mp_mean
        I.m = update_m_P_t1_t_o2(I.H, I.J, I.m_p)
        I.D = update_D_P_t1_t_o2(I.H, I.J, I.m, I.m_p)
        DH = m_mean - I.m
        DJ = D_mean - I.D
        # We impose a bound on H to avoid
        DH[I.H > 5] = np.clip(DH[I.H > 5], -np.inf, 0)
        # divergence of Boltzmann learning
        DH[I.H < -5] = np.clip(DH[I.H < -5], 0, np.inf)
        I.J += etaJ * DJ
        I.H += etaH * DH
        error_H = np.mean(DH**2)
        error_J = np.mean(DJ**2)
        print('P_t1_t_o2', beta_ref, rep, nsf(error_H), nsf(error_J),
              nsf(np.mean((H - I.H)**2)), nsf(np.mean((J - I.J)**2)))
        rep += 1
        if rep >= max_rep or ((error_H < ref_error) and (error_J < ref_error)):
            cond = False
    HP_t1_t = I.H.copy()
    JP_t1_t = I.J.copy()
    time_P_t1_t = time.perf_counter() - time_start


# Run Plefka[t], order 2

    time_start = time.perf_counter()
    I = mf_ising(size)
    I.H = np.zeros(size)
    I.J = np.zeros((size, size))
    rep = 0
    cond = True
    while cond:
        I.m_p = mp_mean
        I.C_p = Cp_mean
        I.m = update_m_P_t_o2(I.H, I.J, I.m_p, I.C_p)
        I.D = update_D_P_t_o2(I.H, I.J, I.m, I.m_p, I.C_p)
        DH = m_mean - I.m
        DJ = D_mean - I.D
        I.J += etaJ * DJ
        I.H += etaH * DH
        error_H = np.mean(DH**2)
        error_J = np.mean(DJ**2)
        print('P_t_o2', beta_ref, rep, nsf(error_H), nsf(error_J),
              nsf(np.mean((H - I.H)**2)), nsf(np.mean((J - I.J)**2)))
        rep += 1
        if rep >= max_rep or ((error_H < ref_error) and (error_J < ref_error)):
            cond = False
    HP_t = I.H.copy()
    JP_t = I.J.copy()
    time_P_t = time.perf_counter() - time_start

    # Run Plefka2[t], order 2
    time_start = time.perf_counter()
    I = mf_ising(size)
    I.H = np.zeros(size)
    I.J = np.zeros((size, size))
    rep = 0
    cond = True
    while cond:
        I.m_p = mp_mean
        I.C_p = Cp_mean
        I.D_p = Dp_mean
        I.m, I.D = update_D_P2_t_o2(I.H, I.J, I.m_p, I.C_p, I.D_p)
        DH = m_mean - I.m
        DJ = D_mean - I.D
        I.J += etaJ * DJ
        I.H += etaH * DH
        error_H = np.mean(DH**2)
        error_J = np.mean(DJ**2)
        print('P2_t_o2', beta_ref, rep, nsf(error_H), nsf(error_J),
              nsf(np.mean((H - I.H)**2)), nsf(np.mean((J - I.J)**2)))
        rep += 1
        if rep >= max_rep or ((error_H < ref_error) and (error_J < ref_error)):
            cond = False
    HP2_t = I.H.copy()
    JP2_t = I.J.copy()
    time_P2_t = time.perf_counter() - time_start

    # Run Plefka[t-1], order 1
    time_start = time.perf_counter()
    I = mf_ising(size)
    I.H = np.zeros(size)
    I.J = np.zeros((size, size))
    rep = 0
    cond = True
    while cond:
        I.m_p = mp_mean
        I.C_p = Cp_mean
        I.m = update_m_P_t1_o1(I.H, I.J, I.m_p)
        I.D = update_D_P_t1_o1(I.H, I.J, I.m_p, I.C_p)
        DH = m_mean - I.m
        DJ = D_mean - I.D
        I.J += etaJ * DJ
        I.H += etaH * DH
        error_H = np.mean(DH**2)
        error_J = np.mean(DJ**2)
        print('P_t1_o1', beta_ref, rep, nsf(error_H), nsf(error_J),
              nsf(np.mean((H - I.H)**2)), nsf(np.mean((J - I.J)**2)))
        rep += 1
        if rep >= max_rep or ((error_H < ref_error) and (error_J < ref_error)):
            cond = False
    HP_t1 = I.H.copy()
    JP_t1 = I.J.copy()
    time_P_t1 = time.perf_counter() - time_start

    filename = 'data/inverse/inverse_' + \
        str(int(beta_ref * 100)) + '_R_' + str(R) + '.npz'
    print(filename)
    np.savez_compressed(filename,
                        H=H, J=J,
                        HP_t1_t=HP_t1_t, HP_t=HP_t, HP_t1=HP_t1, HP2_t=HP2_t,
                        JP_t1_t=JP_t1_t, JP_t=JP_t, JP_t1=JP_t1, JP2_t=JP2_t,
                        time_P_t1_t=time_P_t1_t,
                        time_P_t=time_P_t,
                        time_P_t1=time_P_t1,
                        time_P2_t=time_P2_t)
