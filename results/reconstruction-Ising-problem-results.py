#!/usr/bin/env python3
"""
GPLv3 2020 Miguel Aguilera

This code displays the results of the reconstruction Ising problem
computed from running "generate_data.py",  "inverse-Ising-problem.py"
and "reconstruction-Ising-problem.py"
"""

import context
from plefka import mf_ising
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

plt.rc('text', usetex=True)
font = {'size': 15}
plt.rc('font', **font)
plt.rc('legend', **{'fontsize': 18})


size = 512
R = 1000000
R1 = 100000
J0 = 1.0
H0 = 0.5
Js = 0.1

T = 128
iu1 = np.triu_indices(size, 1)

offset = np.arange(3)
width = 0.20

B = 201
betas = 1 + np.linspace(-1, 1, B) * 0.3

N = len(betas)
M = 4
errorm = np.zeros((M, N))
errorC = np.zeros((M, N))
errorD = np.zeros((M, N))
errorH = np.zeros((M, N))
errorJ = np.zeros((M, N))

mPexp = np.zeros(B)
mP_t1_t = np.zeros(B)
mP_t = np.zeros(B)
mP_t1 = np.zeros(B)
mP2_t = np.zeros(B)
mP_t1_t_0 = np.zeros(B)
mP_t_0 = np.zeros(B)
mP_t1_0 = np.zeros(B)
mP2_t_0 = np.zeros(B)
emP_t1_t = np.zeros(B)
emP_t = np.zeros(B)
emP_t1 = np.zeros(B)
emP2_t = np.zeros(B)

CPexp = np.zeros(B)
CP_t1_t = np.zeros(B)
CP_t = np.zeros(B)
CP_t1 = np.zeros(B)
CP2_t = np.zeros(B)
CP_t1_t_0 = np.zeros(B)
CP_t_0 = np.zeros(B)
CP_t1_0 = np.zeros(B)
CP2_t_0 = np.zeros(B)
eCP_t1_t = np.zeros(B)
eCP_t = np.zeros(B)
eCP_t1 = np.zeros(B)
eCP2_t = np.zeros(B)

DPexp = np.zeros(B)
DP_t1_t = np.zeros(B)
DP_t = np.zeros(B)
DP_t1 = np.zeros(B)
DP2_t = np.zeros(B)
DP_t1_t_0 = np.zeros(B)
DP_t_0 = np.zeros(B)
DP_t1_0 = np.zeros(B)
DP2_t_0 = np.zeros(B)
eDP_t1_t = np.zeros(B)
eDP_t = np.zeros(B)
eDP_t1 = np.zeros(B)
eDP2_t = np.zeros(B)

sigmaPexp = np.zeros(B)
sigmaP_t1_t = np.zeros(B)
sigmaP_t = np.zeros(B)
sigmaP_t1 = np.zeros(B)
sigmaP2_t = np.zeros(B)

sigmaP_t1_t_0 = np.zeros(B)
sigmaP_t_0 = np.zeros(B)
sigmaP_t1_0 = np.zeros(B)
sigmaP2_t_0 = np.zeros(B)

filename_exp = 'data/reconstruction/data-transition-H0-' + str(H0) + '-J0-' + str(
    J0) + '-Js-' + str(Js) + '-N-' + str(size) + '-R-' + str(R1) + '-B-' + str(B) + '.npz'
data_exp = np.load(filename_exp)
print(list(data_exp.keys()))
mPexp = data_exp['m_mean']
CPexp = data_exp['C_mean']
DPexp = data_exp['D_mean']
sigmaPexp = np.exp(data_exp['sigma'])

filename_inv = 'data/inverse/inverse_100_R_' + str(R) + '.npz'
data_inv = np.load(filename_inv)

J = data_inv['J']
JP_t1_t = data_inv['JP_t1_t']
JP_t = data_inv['JP_t']
JP_t1 = data_inv['JP_t1']
JP2_t = data_inv['JP2_t']

for ib in range(len(betas)):
    beta_ref = round(betas[ib], 4)
    print(ib, beta_ref)
    filename_r = 'data/reconstruction/transition_r_' + \
        str(int(round(beta_ref * 1000))) + '_R_' + str(R) + '.npz'
    filename_d = 'data/reconstruction/transition_f_' + \
        str(int(round(beta_ref * 1000))) + '_R_' + str(R) + '.npz'

    data_r = np.load(filename_r)
    data_d = np.load(filename_d)

    mP_t1_t_all = data_r['mP_t1_t']
    mP_t_all = data_r['mP_t']
    mP2_t_all = data_r['mP2_t']
    mP_t1_all = data_r['mP_t1']
    CP_t1_t_all = data_r['CP_t1_t']
    CP_t_all = data_r['CP_t']
    CP2_t_all = data_r['CP2_t']
    CP_t1_all = data_r['CP_t1']
    DP_t1_t_all = data_r['DP_t1_t']
    DP_t_all = data_r['DP_t']
    DP2_t_all = data_r['DP2_t']
    DP_t1_all = data_r['DP_t1']

    mP_t1_t_all_0 = data_d['mP_t1_t']
    mP_t_all_0 = data_d['mP_t']
    mP2_t_all_0 = data_d['mP2_t']
    mP_t1_all_0 = data_d['mP_t1']
    CP_t1_t_all_0 = data_d['CP_t1_t']
    CP_t_all_0 = data_d['CP_t']
    CP2_t_all_0 = data_d['CP2_t']
    CP_t1_all_0 = data_d['CP_t1']
    DP_t1_t_all_0 = data_d['DP_t1_t']
    DP_t_all_0 = data_d['DP_t']
    DP2_t_all_0 = data_d['DP2_t']
    DP_t1_all_0 = data_d['DP_t1']

    sigmaP_t1_t[ib] = np.exp(
        np.sum(JP_t1_t * beta_ref * (DP_t1_t_all - DP_t1_t_all.T)))
    sigmaP_t[ib] = np.exp(np.sum(JP_t * beta_ref * (DP_t_all - DP_t_all.T)))
    sigmaP_t1[ib] = np.exp(
        np.sum(JP_t1 * beta_ref * (DP_t1_all - DP_t1_all.T)))
    sigmaP2_t[ib] = np.exp(
        np.sum(JP2_t * beta_ref * (DP2_t_all - DP2_t_all.T)))

    sigmaP_t1_t_0[ib] = np.exp(
        np.sum(J * beta_ref * (DP_t1_t_all_0 - DP_t1_t_all_0.T)))
    sigmaP_t_0[ib] = np.exp(np.sum(J * beta_ref * (DP_t_all_0 - DP_t_all_0.T)))
    sigmaP_t1_0[ib] = np.exp(
        np.sum(J * beta_ref * (DP_t1_all_0 - DP_t1_all_0.T)))
    sigmaP2_t_0[ib] = np.exp(
        np.sum(J * beta_ref * (DP2_t_all_0 - DP2_t_all_0.T)))

    mP_t1_t[ib] = np.mean(mP_t1_t_all)
    mP_t[ib] = np.mean(mP_t_all)
    mP_t1[ib] = np.mean(mP_t1_all)
    mP2_t[ib] = np.mean(mP2_t_all)
    CP_t1_t[ib] = np.mean(CP_t1_t_all)
    CP_t[ib] = np.mean(CP_t_all)
    CP_t1[ib] = np.mean(CP_t1_all)
    CP2_t[ib] = np.mean(CP2_t_all)
    DP_t1_t[ib] = np.mean(DP_t1_t_all)
    DP_t[ib] = np.mean(DP_t_all)
    DP_t1[ib] = np.mean(DP_t1_all)
    DP2_t[ib] = np.mean(DP2_t_all)

    mP_t1_t_0[ib] = np.mean(mP_t1_t_all_0)
    mP_t_0[ib] = np.mean(mP_t_all_0)
    mP_t1_0[ib] = np.mean(mP_t1_all_0)
    mP2_t_0[ib] = np.mean(mP2_t_all_0)
    CP_t1_t_0[ib] = np.mean(CP_t1_t_all_0)
    CP_t_0[ib] = np.mean(CP_t_all_0)
    CP_t1_0[ib] = np.mean(CP_t1_all_0)
    CP2_t_0[ib] = np.mean(CP2_t_all_0)
    DP_t1_t_0[ib] = np.mean(DP_t1_t_all_0)
    DP_t_0[ib] = np.mean(DP_t_all_0)
    DP_t1_0[ib] = np.mean(DP_t1_all_0)
    DP2_t_0[ib] = np.mean(DP2_t_all_0)
    del data_r
    del data_d


labels = [
    r'Plefka[$t-1,t$]',
    r'Plefka[$t$]',
    r'Plefka[$t-1$]',
    r'Plefka2[$t$]',
    r'Original']
line = [(5, 4), (5, 4), '', '']
cmap = cm.get_cmap('plasma_r')
colors = []
for i in range(4):
    colors += [cmap((i) / 3)]
lws = [2, 2, 2, 3, 1.5]
pos_l = [-0.2, 1.0]

letters = ['A', 'B', 'C', 'D', 'E', 'F']

fig, ax = plt.subplots(2, 3, figsize=(16, 10 * 2 / 3), dpi=300)

nrow = 0
ncol = 0
ax[nrow, ncol].plot(betas, CP_t1_0, dashes=line[2], color=colors[2], lw=lws[2])
ax[nrow, ncol].plot(betas, CP2_t_0, dashes=line[3], color=colors[3], lw=lws[3])
ax[nrow, ncol].plot(betas, CP_t1_t_0, dashes=line[0],
                    color=colors[0], lw=lws[0])
ax[nrow, ncol].plot(betas, CP_t_0, dashes=line[1], color=colors[1], lw=lws[1])
ax[nrow, ncol].plot(betas, CPexp, color='k')
ax[nrow, ncol].plot(betas[np.argmax(CPexp)], [0.021], '*', ms=10, color='k')
ax[nrow, ncol].plot(betas[np.argmax(CP_t1_t_0)], [
                    0.021], '*', ms=10, color=colors[0])
ax[nrow, ncol].plot(betas[np.argmax(CP_t_0)], [0.024],
                    '*', ms=10, color=colors[1])
ax[nrow, ncol].plot(betas[np.argmax(CP_t1_0)], [0.0225],
                    '*', ms=10, color=colors[2])
ax[nrow, ncol].plot(betas[np.argmax(CP2_t_0)], [0.0225],
                    '*', ms=10, color=colors[3])
ax[nrow, ncol].plot([1, 1], [0, 0.025], lw=0.5, color='k')
ax[nrow, ncol].axis([np.min(betas), np.max(betas), 0, 0.025])
ax[nrow, ncol].set_xlabel(r'$\beta / \beta_c$', fontsize=18)
ax[nrow, ncol].set_ylabel(
    r'$\langle C_{ik,t} \rangle$', fontsize=18, rotation=0, labelpad=25)
ax[nrow,
    ncol].text(pos_l[0],
               pos_l[1],
               r'\textbf ' + letters[0],
               transform=ax[nrow,
                            ncol].transAxes,
               fontsize=20,
               va='top',
               ha='right')

nrow = 0
ncol = 1
ax[nrow, ncol].plot(betas, DP_t1_0, dashes=line[2], color=colors[2], lw=lws[2])
ax[nrow, ncol].plot(betas, DP2_t_0, dashes=line[3], color=colors[3], lw=lws[3])
ax[nrow, ncol].plot(betas, DP_t1_t_0, dashes=line[0],
                    color=colors[0], lw=lws[0])
ax[nrow, ncol].plot(betas, DP_t_0, dashes=line[1], color=colors[1], lw=lws[1])
ax[nrow, ncol].plot(betas, DPexp, color='k')
ax[nrow, ncol].plot(betas[np.argmax(DPexp)], [0.021], '*', ms=10, color='k')
ax[nrow, ncol].plot(betas[np.argmax(DP_t1_t_0)], [
                    0.021], '*', ms=10, color=colors[0])
ax[nrow, ncol].plot(betas[np.argmax(DP_t_0)], [0.024],
                    '*', ms=10, color=colors[1])
ax[nrow, ncol].plot(betas[np.argmax(DP_t1_0)], [0.021],
                    '*', ms=10, color=colors[2])
ax[nrow, ncol].plot(betas[np.argmax(DP2_t_0)], [0.0225],
                    '*', ms=10, color=colors[3])
ax[nrow, ncol].plot([1, 1], [0, 0.025], lw=0.5, color='k')
ax[nrow, ncol].axis([np.min(betas), np.max(betas), 0, 0.025])
ax[nrow, ncol].set_xlabel(r'$\beta / \beta_c$', fontsize=18)
ax[nrow, ncol].set_ylabel(
    r'$\langle D_{il,t} \rangle$', fontsize=18, rotation=0, labelpad=25)
ax[nrow,
    ncol].text(pos_l[0],
               pos_l[1],
               r'\textbf ' + letters[1],
               transform=ax[nrow,
                            ncol].transAxes,
               fontsize=20,
               va='top',
               ha='right')

nrow = 0
ncol = 2

ax[nrow, ncol].plot(betas, sigmaPexp, color='k', label=labels[4])
ax[nrow, ncol].plot(betas, sigmaP_t1_t_0, dashes=line[0],
                    color=colors[0], lw=lws[0], label=labels[0])
ax[nrow, ncol].plot(betas, sigmaP_t_0, dashes=line[1],
                    color=colors[1], lw=lws[1], label=labels[1])
ax[nrow, ncol].plot(betas, sigmaP_t1_0, dashes=line[2],
                    color=colors[2], lw=lws[2], label=labels[2])
ax[nrow, ncol].plot(betas, sigmaP2_t_0, dashes=line[3],
                    color=colors[3], lw=lws[3], label=labels[3])
ax[nrow, ncol].plot(betas, sigmaPexp, color='k')
ax[nrow, ncol].plot(betas, sigmaP_t1_t_0, dashes=line[0],
                    color=colors[0], lw=lws[0])
ax[nrow, ncol].plot(betas, sigmaP_t1_0, dashes=line[2],
                    color=colors[2], lw=lws[2])
ax[nrow, ncol].plot(betas[np.argmax(sigmaPexp)], [120], '*', ms=10, color='k')
ax[nrow, ncol].plot(betas[np.argmax(sigmaP_t1_t_0)], [
                    130], '*', ms=10, color=colors[0])
ax[nrow, ncol].plot(betas[np.argmax(sigmaP_t_0)], [
                    130], '*', ms=10, color=colors[1])
ax[nrow, ncol].plot(betas[np.argmax(sigmaP_t1_0)], [
                    110], '*', ms=10, color=colors[2])
ax[nrow, ncol].plot(betas[np.argmax(sigmaP2_t_0)], [
                    110], '*', ms=10, color=colors[3])
ax[nrow, ncol].plot([1, 1], [0, 135], lw=0.5, color='k')
ax[nrow, ncol].axis([np.min(betas), np.max(betas), 0, 135])
ax[nrow, ncol].set_xlabel(r'$\beta / \beta_c$', fontsize=18)
ax[nrow, ncol].set_ylabel(
    r'$\mathrm{e}^{\langle \sigma_t\rangle}$', fontsize=18, rotation=0, labelpad=25)
ax[nrow,
    ncol].text(pos_l[0],
               pos_l[1],
               r'\textbf ' + letters[2],
               transform=ax[nrow,
                            ncol].transAxes,
               fontsize=20,
               va='top',
               ha='right')

nrow = 1
ncol = 0
ax[nrow, ncol].plot(betas, CP_t1, dashes=line[2], color=colors[2], lw=lws[2])
ax[nrow, ncol].plot(betas, CP2_t, dashes=line[3], color=colors[3], lw=lws[3])
ax[nrow, ncol].plot(betas, CP_t1_t, dashes=line[0], color=colors[0], lw=lws[0])
ax[nrow, ncol].plot(betas, CP_t, dashes=line[1], color=colors[1], lw=lws[1])
ax[nrow, ncol].plot(betas, CPexp, color='k')
ax[nrow, ncol].plot(betas[np.argmax(CPexp)], [0.0225], '*', ms=10, color='k')
ax[nrow, ncol].plot(betas[np.argmax(CP_t1_t)], [0.0225],
                    '*', ms=10, color=colors[0])
ax[nrow, ncol].plot(betas[np.argmax(CP_t)], [0.021],
                    '*', ms=10, color=colors[1])
ax[nrow, ncol].plot(betas[np.argmax(CP_t1)], [0.0225],
                    '*', ms=10, color=colors[2])
ax[nrow, ncol].plot(betas[np.argmax(CP2_t)], [0.024],
                    '*', ms=10, color=colors[3])
ax[nrow, ncol].plot([1, 1], [0, 0.025], lw=0.5, color='k')
ax[nrow, ncol].axis([np.min(betas), np.max(betas), 0, 0.025])
ax[nrow, ncol].set_xlabel(r'$\tilde\beta$', fontsize=18)
ax[nrow, ncol].set_ylabel(
    r'$\langle C_{ik,t} \rangle$', fontsize=18, rotation=0, labelpad=25)
ax[nrow,
    ncol].text(pos_l[0],
               pos_l[1],
               r'\textbf ' + letters[3],
               transform=ax[nrow,
                            ncol].transAxes,
               fontsize=20,
               va='top',
               ha='right')

nrow = 1
ncol = 1
ax[nrow, ncol].plot(betas, DP_t1, dashes=line[2], color=colors[2], lw=lws[2])
ax[nrow, ncol].plot(betas, DP2_t, dashes=line[3], color=colors[3], lw=lws[3])
ax[nrow, ncol].plot(betas, DP_t1_t, dashes=line[0], color=colors[0], lw=lws[0])
ax[nrow, ncol].plot(betas, DP_t, dashes=line[1], color=colors[1], lw=lws[1])
ax[nrow, ncol].plot(betas, DPexp, color='k')
ax[nrow, ncol].plot(betas[np.argmax(DPexp)], [0.0225], '*', ms=10, color='k')
ax[nrow, ncol].plot(betas[np.argmax(DP_t1_t)], [0.0225],
                    '*', ms=10, color=colors[0])
ax[nrow, ncol].plot(betas[np.argmax(DP_t)], [0.021],
                    '*', ms=10, color=colors[1])
ax[nrow, ncol].plot(betas[np.argmax(DP_t1)], [0.0225],
                    '*', ms=10, color=colors[2])
ax[nrow, ncol].plot(betas[np.argmax(DP2_t)], [0.024],
                    '*', ms=10, color=colors[3])
ax[nrow, ncol].plot([1, 1], [0, 0.025], lw=0.5, color='k')
ax[nrow, ncol].axis([np.min(betas), np.max(betas), 0, 0.025])
ax[nrow, ncol].set_xlabel(r'$\tilde\beta$', fontsize=18)
ax[nrow, ncol].set_ylabel(
    r'$\langle D_{il,t} \rangle$', fontsize=18, rotation=0, labelpad=25)
ax[nrow,
    ncol].text(pos_l[0],
               pos_l[1],
               r'\textbf ' + letters[4],
               transform=ax[nrow,
                            ncol].transAxes,
               fontsize=20,
               va='top',
               ha='right')

nrow = 1
ncol = 2
ax[nrow, ncol].plot(betas, sigmaP_t1, dashes=line[2],
                    color=colors[2], lw=lws[2])
ax[nrow, ncol].plot(betas, sigmaP2_t, dashes=line[3],
                    color=colors[3], lw=lws[3])
ax[nrow, ncol].plot(betas, sigmaP_t1_t, dashes=line[0],
                    color=colors[0], lw=lws[0])
ax[nrow, ncol].plot(betas, sigmaP_t, dashes=line[1],
                    color=colors[1], lw=lws[1])
ax[nrow, ncol].plot(betas, sigmaPexp, color='k')
ax[nrow, ncol].plot(betas[np.argmax(sigmaPexp)], [110], '*', ms=10, color='k')
ax[nrow, ncol].plot(betas[np.argmax(sigmaP_t1_t)], [
                    120], '*', ms=10, color=colors[0])
ax[nrow, ncol].plot(betas[np.argmax(sigmaP_t)], [
                    100], '*', ms=10, color=colors[1])
ax[nrow, ncol].plot(betas[np.argmax(sigmaP_t1)], [
                    130], '*', ms=10, color=colors[2])
ax[nrow, ncol].plot(betas[np.argmax(sigmaP2_t)], [
                    120], '*', ms=10, color=colors[3])
ax[nrow, ncol].plot([1, 1], [0, 135], lw=0.5, color='k')
ax[nrow, ncol].axis([np.min(betas), np.max(betas), 0, 135])
ax[nrow, ncol].set_xlabel(r'$\tilde\beta$', fontsize=18)
ax[nrow, ncol].set_ylabel(
    r'$\mathrm{e}^{\langle \sigma_t\rangle}$', fontsize=18, rotation=0, labelpad=25)
ax[nrow,
    ncol].text(pos_l[0],
               pos_l[1],
               r'\textbf ' + letters[5],
               transform=ax[nrow,
                            ncol].transAxes,
               fontsize=20,
               va='top',
               ha='right')
plt.figlegend(
    loc='upper center',
    bbox_to_anchor=(
        0.5,
        1.),
    borderaxespad=0,
    ncol=5)
fig.tight_layout(h_pad=0.3, w_pad=0.7, rect=[0, 0, 1, 0.95])
plt.savefig('img/results-reconstruted-transition.pdf', bbox_inches='tight')
