#!/usr/bin/env python3
"""
GPLv3 2020 Miguel Aguilera

This code displays the results of the forward Ising problem computed
from running "generate_data.py" and "forward-Ising-problem.py"
"""

from mf_ising import mf_ising
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

plt.rc('text', usetex=True)
font = {'size': 15}
plt.rc('font', **font)
plt.rc('legend', **{'fontsize': 18})


size = 512
beta = 1.0
R = 1000000

T = 128

cmap = cm.get_cmap('plasma_r')
colors = []
for i in range(4):
    colors += [cmap((i) / 3)]
line = ['--', '--', '-', '-']
line = [(5, 4), (5, 4), '', '']

labels = [
    r'Plefka[$t-1,t$]',
    r'Plefka[$t$]',
    r'Plefka[$t-1$]',
    r'Plefka2[$t$]',
    r'Original']
lws = [2, 2, 2, 3, 1.5]

filename = 'data/forward/forward_100_R_' + str(R) + '.npz'
data = np.load(filename)

m_exp = data['m_exp']
C_exp = data['C_exp']
D_exp = data['D_exp']

time_P_t1_t = data['time_P_t1_t']
time_P_t = data['time_P_t']
time_P_t1 = data['time_P_t1']
time_P2_t = data['time_P2_t']


mP_t1_t_mean = data['mP_t1_t_mean']
mP_t_mean = data['mP_t_mean']
mP_t1_mean = data['mP_t1_mean']
mP2_t_mean = data['mP2_t_mean']
CP_t1_t_mean = data['CP_t1_t_mean']
CP_t_mean = data['CP_t_mean']
CP_t1_mean = data['CP_t1_mean']
CP2_t_mean = data['CP2_t_mean']
DP_t1_t_mean = data['DP_t1_t_mean']
DP_t_mean = data['DP_t_mean']
DP_t1_mean = data['DP_t1_mean']
DP2_t_mean = data['DP2_t_mean']

mPexp_mean = data['mPexp_mean']
CPexp_mean = data['CPexp_mean']
DPexp_mean = data['DPexp_mean']

mP_t1_t = data['mP_t1_t']
mP_t = data['mP_t']
mP_t1 = data['mP_t1']
mP2_t = data['mP2_t']
CP_t1_t = data['CP_t1_t']
CP_t = data['CP_t']
CP_t1 = data['CP_t1']
CP2_t = data['CP2_t']
DP_t1_t = data['DP_t1_t']
DP_t = data['DP_t']
DP_t1 = data['DP_t1']
DP2_t = data['DP2_t']

EmP_t1_t = data['EmP_t1_t']
EmP_t = data['EmP_t']
EmP_t1 = data['EmP_t1']
EmP2_t = data['EmP2_t']
ECP_t1_t = data['ECP_t1_t']
ECP_t = data['ECP_t']
ECP_t1 = data['ECP_t1']
ECP2_t = data['ECP2_t']
EDP_t1_t = data['EDP_t1_t']
EDP_t = data['EDP_t']
EDP_t1 = data['EDP_t1']
EDP2_t = data['EDP2_t']


steps = np.arange(T + 1)
iu1 = np.triu_indices(size, 1)

B = 21
betas = 1 + np.linspace(-1, 1, B) * 0.3


M = 4
N = len(betas)
errorm = np.zeros((M, N))
errorC = np.zeros((M, N))
errorD = np.zeros((M, N))

for ind in range(len(betas)):
    beta_ref = round(betas[ind], 3)
    filename = 'data/forward/forward_' + \
        str(int(beta_ref * 100)) + '_R_' + str(R) + '.npz'
    data = np.load(filename)

    EmP_t1_t = np.mean(data['EmP_t1_t'])
    EmP_t = np.mean(data['EmP_t'])
    EmP_t1 = np.mean(data['EmP_t1'])
    EmP2_t = np.mean(data['EmP2_t'])
    ECP_t1_t = np.mean(data['ECP_t1_t'])
    ECP_t = np.mean(data['ECP_t'])
    ECP_t1 = np.mean(data['ECP_t1'])
    ECP2_t = np.mean(data['ECP2_t'])
    EDP_t1_t = np.mean(data['EDP_t1_t'])
    EDP_t = np.mean(data['EDP_t'])
    EDP_t1 = np.mean(data['EDP_t1'])
    EDP2_t = np.mean(data['EDP2_t'])

    errorm[:, ind] = np.array([EmP_t1_t, EmP_t, EmP_t1, EmP2_t])
    errorC[:, ind] = np.array([ECP_t1_t, ECP_t, ECP_t1, ECP2_t])
    errorD[:, ind] = np.array([EDP_t1_t, EDP_t, EDP_t1, EDP2_t])


letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
pos_l = [-0.2, 1.0]

fig, ax = plt.subplots(3, 3, figsize=(16, 10), dpi=300)

ax[0, 0].plot(steps[0:1], mPexp_mean[0:1], color='k', label=labels[4])
ax[0, 0].plot(steps, mP_t1_t_mean, dashes=line[0],
              color=colors[0], lw=lws[0], label=labels[0])
ax[0, 0].plot(steps, mP_t_mean, dashes=line[1],
              color=colors[1], lw=lws[1], label=labels[1])
ax[0, 0].plot(steps, mP_t1_mean, dashes=line[2],
              color=colors[2], lw=lws[2], label=labels[2])
ax[0, 0].plot(steps, mP2_t_mean, dashes=line[3],
              color=colors[3], lw=lws[3], label=labels[3])
ax[0, 0].plot(steps, mP_t1_t_mean, dashes=line[0], color=colors[0], lw=lws[0])
ax[0, 0].plot(steps, mP_t_mean, dashes=line[1], color=colors[1], lw=lws[1])
ax[0, 0].plot(steps[0:1], mPexp_mean[0:1], color='k')
ax[0, 0].plot(steps, mPexp_mean, 'k', lw=lws[4])  # , label=r'$P$')
ax[0, 0].set_xlabel(r'$t$', fontsize=18)
ax[0, 0].set_ylabel(r'$\langle m_{i,t} \rangle$',
                    fontsize=18, rotation=0, labelpad=25)
ax[0, 0].text(pos_l[0], pos_l[1], r'\textbf ' + letters[0],
              transform=ax[0, 0].transAxes, fontsize=20, va='top', ha='right')
ax[0, 0].axis([0, T, 0.2, 1])


ax[0, 1].plot(steps, CP_t1_mean, dashes=line[2], color=colors[2], lw=lws[2])
ax[0, 1].plot(steps, CP2_t_mean, dashes=line[3], color=colors[3], lw=lws[3])
ax[0, 1].plot(steps, CP_t1_t_mean, dashes=line[0], color=colors[0], lw=lws[0])
ax[0, 1].plot(steps, CP_t_mean, dashes=line[1], color=colors[1], lw=lws[1])
ax[0, 1].plot(steps, CPexp_mean, 'k', lw=lws[4])
ax[0, 1].set_xlabel(r'$t$', fontsize=18)
ax[0, 1].set_ylabel(r'$\langle C_{ik,t} \rangle$',
                    fontsize=18, rotation=0, labelpad=25)
ax[0, 1].set_yticks([0.00, 0.006, 0.012, 0.018])
ax[0, 1].text(pos_l[0], pos_l[1], r'\textbf ' + letters[1],
              transform=ax[0, 1].transAxes, fontsize=20, va='top', ha='right')
ax[0, 1].axis([0, T, 0, 0.02])

ax[0, 2].plot(steps, DP_t1_mean, dashes=line[2], color=colors[2], lw=lws[2])
ax[0, 2].plot(steps, DP2_t_mean, dashes=line[3], color=colors[3], lw=lws[3])
ax[0, 2].plot(steps, DP_t1_t_mean, dashes=line[0], color=colors[0], lw=lws[0])
ax[0, 2].plot(steps, DP_t_mean, dashes=line[1], color=colors[1], lw=lws[1])
ax[0, 2].plot(steps, DPexp_mean, 'k', lw=lws[4])
ax[0, 2].set_yticks([0.00, 0.006, 0.012, 0.018])
ax[0, 2].set_xlabel(r'$t$', fontsize=18)
ax[0, 2].set_ylabel(r'$\langle D_{il,t} \rangle$',
                    fontsize=18, rotation=0, labelpad=25)
ax[0, 2].text(pos_l[0], pos_l[1], r'\textbf ' + letters[2],
              transform=ax[0, 2].transAxes, fontsize=20, va='top', ha='right')
ax[0, 2].axis([0, T, 0, 0.02])


ax[1, 0].plot(m_exp, mP_t1_t, 'v', color=colors[0], ms=3, rasterized=True)
ax[1, 0].plot(m_exp, mP_t, 's', color=colors[1], ms=3, rasterized=True)
ax[1, 0].plot(m_exp, mP_t1, 'd', color=colors[2], ms=3, rasterized=True)
ax[1, 0].plot(m_exp, mP2_t, 'o', color=colors[3], ms=3, rasterized=True)
ax[1, 0].plot(sorted(m_exp), sorted(m_exp), 'k', lw=2.5, rasterized=True)
ax[1, 0].set_xlabel(r'$\mathbf{m}^o$', fontsize=18)
ax[1, 0].set_ylabel(r'$\mathbf{m}^p$', fontsize=18, rotation=0, labelpad=15)
ax[1, 0].text(pos_l[0], pos_l[1], r'\textbf ' + letters[3],
              transform=ax[1, 0].transAxes, fontsize=20, va='top', ha='right')


ax[1, 1].plot(C_exp[iu1], CP_t1_t[iu1], 'v',
              color=colors[0], ms=3, rasterized=True)
ax[1, 1].plot(C_exp[iu1], CP_t[iu1], 's',
              color=colors[1], ms=3, rasterized=True)
ax[1, 1].plot(C_exp[iu1], CP_t1[iu1], 'd',
              color=colors[2], ms=3, rasterized=True)
ax[1, 1].plot(C_exp[iu1], CP2_t[iu1], 'o',
              color=colors[3], ms=3, rasterized=True)
ax[1, 1].plot(sorted(C_exp[iu1]), sorted(
    C_exp[iu1]), 'k', lw=2.5, rasterized=True)
ax[1, 1].set_xlabel(r'$\mathbf{C}^o$', fontsize=18)
ax[1, 1].set_ylabel(r'$\mathbf{C}^p$', fontsize=18, rotation=0, labelpad=15)
ax[1, 1].text(pos_l[0], pos_l[1], r'\textbf ' + letters[4],
              transform=ax[1, 1].transAxes, fontsize=20, va='top', ha='right')

ax[1, 2].plot(D_exp.flatten(), DP_t1_t.flatten(), 'v',
              color=colors[0], ms=3, rasterized=True)
ax[1, 2].plot(D_exp.flatten(), DP_t.flatten(), 's',
              color=colors[1], ms=3, rasterized=True)
ax[1, 2].plot(D_exp.flatten(), DP_t1.flatten(), 'd',
              color=colors[2], ms=3, rasterized=True)
ax[1, 2].plot(D_exp.flatten(), DP2_t.flatten(), 'o',
              color=colors[3], ms=3, rasterized=True)
ax[1, 2].plot(sorted(D_exp.flatten()), sorted(
    D_exp.flatten()), 'k', lw=2.5, rasterized=True)
ax[1, 2].set_xlabel(r'$\mathbf{D}^o$', fontsize=18)
ax[1, 2].set_ylabel(r'$\mathbf{D}^p$', fontsize=18, rotation=0, labelpad=15)
ax[1, 2].text(pos_l[0], pos_l[1], r'\textbf ' + letters[5],
              transform=ax[1, 2].transAxes, fontsize=20, va='top', ha='right')


for m in range(M):
    ax[2, 0].semilogy(betas, errorm[m, :], dashes=line[m],
                      color=colors[m], lw=lws[m])  # , label=labels[m])
for m in range(2):
    ax[2, 0].semilogy(betas, errorm[m, :], dashes=line[m],
                      color=colors[m], lw=lws[m])
ax[2, 0].set_xlabel(r'$\beta/\beta_c$', fontsize=18)
ax[2, 0].set_ylabel(r'$\epsilon_m$', fontsize=18, rotation=0, labelpad=15)
ax[2, 0].text(pos_l[0], pos_l[1], r'\textbf ' + letters[6],
              transform=ax[2, 0].transAxes, fontsize=20, va='top', ha='right')

for m in range(M):
    ax[2, 1].semilogy(betas, errorC[m, :], dashes=line[m],
                      color=colors[m], lw=lws[m])  # , label=labels[m])
for m in range(2):
    ax[2, 1].semilogy(betas, errorC[m, :], dashes=line[m],
                      color=colors[m], lw=lws[m])
ax[2, 1].set_xlabel(r'$\beta/\beta_c$', fontsize=18)
ax[2, 1].set_ylabel(r'$\epsilon_C$', fontsize=18, rotation=0, labelpad=15)
ax[2, 1].text(pos_l[0], pos_l[1], r'\textbf ' + letters[7],
              transform=ax[2, 1].transAxes, fontsize=20, va='top', ha='right')

for m in range(M):
    ax[2, 2].semilogy(betas, errorD[m, :], dashes=line[m],
                      color=colors[m], lw=lws[m])  # , label=labels[m])
for m in range(2):
    ax[2, 2].semilogy(betas, errorD[m, :], dashes=line[m],
                      color=colors[m], lw=lws[m])
ax[2, 2].set_xlabel(r'$\beta/\beta_c$', fontsize=18)
ax[2, 2].set_ylabel(r'$\epsilon_D$', fontsize=18, rotation=0, labelpad=15)
ax[2, 2].text(pos_l[0], pos_l[1], r'\textbf ' + letters[8],
              transform=ax[2, 2].transAxes, fontsize=20, va='top', ha='right')

plt.figlegend(
    loc='upper center',
    bbox_to_anchor=(
        0.5,
        1.),
    borderaxespad=0,
    ncol=5)
fig.tight_layout(h_pad=0.3, w_pad=0.7, rect=[0, 0, 1, 0.975])
plt.savefig('img/results-forward-Ising.pdf', bbox_inches='tight')
