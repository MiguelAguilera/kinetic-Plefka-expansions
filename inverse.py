#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:45:08 2018

@author: maguilera
"""

from mf_ising import mf_ising
from ising_functions import update_m_P_t1_t_o2, update_D_P_t1_t_o2, update_m_P_t_o2, update_D_P_t_o2
from ising_functions import update_m_P_t1_o1, update_D_P_t1_o1, update_D_P2_t_o2
import numpy as np
import time
from sys import argv

def nsf(num, n=4):
    """n-Significant Figures"""
    numstr = ("{0:.%ie}" % (n-1)).format(num)
    return float(numstr)

def main():

    if len(argv) < 2:
        print("Usage: " + argv[0] + " <beta_ref>" )
        exit(1)


    gamma1 = 0.5
    gamma2 = 0.1
    if gamma1 == 0.5 and gamma2 == 0.1:
        beta0 = 1.1123
    beta_ref = float(argv[1])        # beta / beta_c value


    size = 512
    R = 500000
    mode = 'c'

    T = 128

    max_rep = 200

    timesteps0 = np.arange(4, T - 2)
    T0=len(timesteps0)


    filename = 'data/m-c-ts0-gamma1-' + str(gamma1) + '-gamma2-' + str(
        gamma2) + '-s-' + str(size) + '-R-' + str(R) + '-beta-' + str(beta_ref) + '.npz'
    data = np.load(filename)
    H = data['H']
    J = data['J']
    m_exp=data['m']
    C_exp=data['C']
    D_exp=data['D']
    s0 = np.array(data['s0'])
    del data

    # Reference error is defined to be 10 times bigger than variance of noise from truncation to 5 decimal places
    ref_error = 10*1/12*(0.00001/T0)**2
    print('ref error:',ref_error)


    eta = 1 *  0.2
    for phase in range(1):
        shuffled_steps = np.random.permutation(T0)
        if phase==0:
            etaH = eta * 0.2
            etaJ = eta / size**0.5
            rel_error_ref = 0.02
        else:
            etaH = eta * 0.1
            etaJ = eta / size**0.5 * 0.1
            rel_error_ref = 0.02 * 0.1
        
        rel_error_ref = 0.02 * 0.2**(phase)
        
        print('Initiate phase',phase, '- eta =' , eta)
        if phase ==0:
            HP_t1_t_0 = np.arctanh(m_exp[:, T-2])
            JP_t1_t_0 = np.zeros((size,size))
            HP_t_0 = np.arctanh(m_exp[:, T-2])
            JP_t_0 = np.zeros((size,size))
            HP_t1_0 = np.arctanh(m_exp[:, T-2])
            JP_t1_0 = np.zeros((size,size))
            HP2_t_0 = np.arctanh(m_exp[:, T-2])
            JP2_t_0 = np.zeros((size,size))
        else:
            filename = 'data/results/inverse_' + str(int(beta_ref * 100)) +'_R_' + str(R) +'_phase_' + str(phase-1) + '.npz'
            data=np.load(filename)
            HP_t1_t_0 = data['HP_t1_t']
            JP_t1_t_0 = data['JP_t1_t']
            HP_t_0 = data['HP_t']
            JP_t_0 = data['JP_t']
            HP_t1_0 = data['HP_t1']
            JP_t1_0 = data['JP_t1']
            HP2_t_0 = data['HP2_t']
            JP2_t_0 = data['JP2_t']
            del data

        HP_t1_t = HP_t1_t_0.copy()
        JP_t1_t = JP_t1_t_0.copy()
        HP_t = HP_t_0.copy()
        JP_t = JP_t_0.copy()
        HP2_t = HP2_t_0.copy()
        JP2_t = JP2_t_0.copy()
        HP_t1 = HP_t1_0.copy()
        JP_t1 = JP_t1_0.copy()
        # Run Plefka[t-1,t], order 2
        

        time_start = time.perf_counter()
        I = mf_ising(size)
        I.H=HP_t1_t_0.copy()
        I.J=JP_t1_t_0.copy()
        error_H = np.inf
        error_J = np.inf
        rep = 0
        cond = True
        while cond:
            DH_mean = np.zeros(size)
            DJ_mean = np.zeros((size, size))
            error_H_prev = error_H
            error_J_prev = error_J
            DH_mean_v = np.zeros(size)
            DJ_mean_v = np.zeros((size, size))
            for ind in shuffled_steps:
                t=timesteps0[ind]
                I.m_p = m_exp[:, t]
                I.m = update_m_P_t1_t_o2(I.H, I.J, I.m_p)
                I.D = update_D_P_t1_t_o2(I.H, I.J, I.m, I.m_p)
                DJ =  (D_exp[:, :, t + 1] + np.einsum('i,j->ij',m_exp[:, t + 1],I.m_p,optimize=True)) - (I.D + np.einsum('i,j->ij',I.m,I.m_p,optimize=True))
                DH = m_exp[:, t + 1] - I.m
                I.H += etaH * DH 
                I.J += etaJ * DJ 
                DH_mean += etaH * DH / T0
                DJ_mean += etaJ * DJ / T0
            print('P_t1_t_o2', beta_ref, phase,  rep, nsf(np.mean(DH_mean**2)), nsf(np.mean(DJ_mean**2)), nsf(np.mean((H - I.H)**2)), nsf(np.mean((J - I.J)**2)))
            rep += 1
            if rep >= max_rep or np.mean(DH_mean**2)<ref_error or np.mean(DJ_mean**2)<ref_error:# or  rel_d_error_H>0 :
                    cond=False
        HP_t1_t = I.H.copy()
        JP_t1_t = I.J.copy()
        time_P_t1_t = time.perf_counter() - time_start


    # Run Plefka[t], order 2

        time_start = time.perf_counter()
        I = mf_ising(size)
        I.H=HP_t_0.copy()
        I.J=JP_t_0.copy()
        error_H = np.inf
        error_J = np.inf
        rep = 0
        cond = True
        while cond:
            DH_mean = np.zeros(size)
            DJ_mean = np.zeros((size, size))
            error_H_prev = error_H
            error_J_prev = error_J
            DH_mean_v = np.zeros(size)
            DJ_mean_v = np.zeros((size, size))
            for ind in shuffled_steps:
                t=timesteps0[ind]
                I.m_p = m_exp[:, t]
                I.C_p = C_exp[:, :, t]
                I.m = update_m_P_t_o2(I.H, I.J, I.m_p, I.C_p)
                I.D = update_D_P_t_o2(I.H, I.J, I.m, I.m_p, I.C_p)
                DJ =  (D_exp[:, :, t + 1] + np.einsum('i,j->ij',m_exp[:, t + 1],I.m_p,optimize=True)) - (I.D + np.einsum('i,j->ij',I.m,I.m_p,optimize=True))
                DH = m_exp[:, t + 1] - I.m
                I.J += etaJ * DJ
                I.H += etaH * DH
                DH_mean += etaH * DH / T0
                DJ_mean += etaJ * DJ / T0
            print('P_t_o2', beta_ref, phase,  rep, nsf(np.mean(DH_mean**2)), nsf(np.mean(DJ_mean**2)), nsf(np.mean((H - I.H)**2)), nsf(np.mean((J - I.J)**2)))
            rep += 1
            if rep >= max_rep or np.mean(DH_mean**2)<ref_error or np.mean(DJ_mean**2)<ref_error:
                    cond=False
        HP_t = I.H.copy()
        JP_t = I.J.copy()
        time_P_t = time.perf_counter() - time_start



        # Run Plefka2[t], order 2
        time_start = time.perf_counter()
        I = mf_ising(size)
        I.H=HP2_t_0.copy()
        I.J=JP2_t_0.copy()
        error_H = np.inf
        error_J = np.inf
        rep = 0
        cond = True
        while cond:
            DH_mean = np.zeros(size)
            DJ_mean = np.zeros((size, size))
            error_H_prev = error_H
            error_J_prev = error_J
            DH_mean_v = np.zeros(size)
            DJ_mean_v = np.zeros((size, size))
            for ind in shuffled_steps:
                t=timesteps0[ind]
                I.m_p = m_exp[:, t]
                I.C_p = C_exp[:, :, t]
                I.D_p = D_exp[:, :, t]
                I.m, I.D = update_D_P2_t_o2(I.H, I.J, I.m_p, I.C_p, I.D_p)
                DJ =  (D_exp[:, :, t + 1] + np.einsum('i,j->ij',m_exp[:, t + 1],I.m_p,optimize=True)) - (I.D + np.einsum('i,j->ij',I.m,I.m_p,optimize=True))
                DH = m_exp[:, t + 1] - I.m
                I.J += etaJ * DJ
                I.H += etaH * DH
                DH_mean += etaH * DH / T0
                DJ_mean += etaJ * DJ / T0
            print('P2_t_o2', beta_ref, phase,  rep, nsf(np.mean(DH_mean**2)), nsf(np.mean(DJ_mean**2)), nsf(np.mean((H - I.H)**2)), nsf(np.mean((J - I.J)**2)))
            rep += 1
            if rep >= max_rep or np.mean(DH_mean**2)<ref_error or np.mean(DJ_mean**2)<ref_error:
                    cond=False
        HP2_t = I.H.copy()
        JP2_t = I.J.copy()
        time_P2_t = time.perf_counter() - time_start

        filename = 'data/results/inverse0_' + str(int(beta_ref * 100)) +'_R_' + str(R) +'_phase_' + str(phase) + '.npz'
        print(filename)
        np.savez_compressed(filename,
                            H=H, J=J,
                            HP_t1_t=HP_t1_t, HP_t=HP_t, HP2_t=HP2_t,
                            JP_t1_t=JP_t1_t, JP_t=JP_t, JP2_t=JP2_t,
                            time_P_t1_t=time_P_t1_t,
                            time_P_t=time_P_t,
                            time_P2_t=time_P2_t)
            
        # Run Plefka[t-1], order 1
        time_start = time.perf_counter()
        I = mf_ising(size)
        I.H=HP_t1_0.copy()
        I.J=JP_t1_0.copy()
        error_H = np.inf
        error_J = np.inf
        rep = 0
        cond = True
        while cond:
            DH_mean = np.zeros(size)
            DJ_mean = np.zeros((size, size))
            error_H_prev = error_H
            error_J_prev = error_J
            DH_mean_v = np.zeros(size)
            DJ_mean_v = np.zeros((size, size))
            for ind in shuffled_steps:
                t=timesteps0[ind]
                I.m_p = m_exp[:, t]
                I.C_p = C_exp[:, :, t]
                I.m = update_m_P_t1_o1(I.H, I.J, I.m_p)
                I.D = update_D_P_t1_o1(I.H, I.J, I.m_p, I.C_p)
                DJ =  (D_exp[:, :, t + 1] + np.einsum('i,j->ij',m_exp[:, t + 1],I.m_p,optimize=True)) - (I.D + np.einsum('i,j->ij',I.m,I.m_p,optimize=True))
                DH = m_exp[:, t + 1] - I.m
                I.J += etaJ * DJ
                I.H += etaH * DH
                DH_mean += etaH * DH / T0
                DJ_mean += etaJ * DJ / T0
            print('P_t1_o2', beta_ref, phase,  rep, nsf(np.mean(DH_mean**2)), nsf(np.mean(DJ_mean**2)), nsf(np.mean((H - I.H)**2)), nsf(np.mean((J - I.J)**2)))
            rep += 1
            if rep >= max_rep or np.mean(DH_mean**2)<ref_error or np.mean(DJ_mean**2)<ref_error:
                    cond=False
        HP_t1 = I.H.copy()
        JP_t1 = I.J.copy()
        time_P_t1 = time.perf_counter() - time_start
        

        filename = 'data/results/inverse_' + str(int(beta_ref * 100)) +'_R_' + str(R) +'_phase_' + str(phase) + '.npz'
        print(filename)
        np.savez_compressed(filename,
                            H=H, J=J,
                            HP_t1_t=HP_t1_t, HP_t=HP_t, HP_t1=HP_t1, HP2_t=HP2_t,
                            JP_t1_t=JP_t1_t, JP_t=JP_t, JP_t1=JP_t1, JP2_t=JP2_t,
                            time_P_t1_t=time_P_t1_t,
                            time_P_t=time_P_t,
                            time_P_t1=time_P_t1,
                            time_P2_t=time_P2_t)
                            
if __name__ == "__main__":
    main()
