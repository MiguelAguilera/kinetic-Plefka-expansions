#!/usr/bin/env python3
"""
GPLv3 2020 Miguel Aguilera

This code displays the results of the inverse Ising problem computed
from running "generate_data.py" and "inverse-Ising-problem.py"
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


T = 2**7
iu1 = np.triu_indices(size, 1)

offset = np.arange(3)
width = 0.20

beta = 1.0

B = 21
betas = 1 + np.linspace(-1, 1, B) * 0.3

M = 4
errorm = np.zeros((M, B))
errorC = np.zeros((M, B))
errorD = np.zeros((M, B))
errorH = np.zeros((M, B))
errorJ = np.zeros((M, B))


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


for ind in range(len(betas)):

    beta_ref = round(betas[ind], 3)
    print(ind, beta_ref)
    filename = 'data/inverse/inverse_' + \
        str(int(beta_ref * 100)) + '_R_' + str(R) + '.npz'
    data = np.load(filename)
    HP_t1_t = data['HP_t1_t']
    HP_t = data['HP_t']
    HP_t1 = data['HP_t1']
    HP2_t = data['HP2_t']
    JP_t1_t = data['JP_t1_t']
    JP_t = data['JP_t']
    JP_t1 = data['JP_t1']
    JP2_t = data['JP2_t']
    H = data['H']
    J = data['J']
    errorH[:,
           ind] = np.array([np.mean((H - HP_t1_t)**2),
                            np.mean((H - HP_t)**2),
                            np.mean((H - HP_t1)**2),
                            np.mean((H - HP2_t)**2)])
    errorJ[:,
           ind] = np.array([np.mean((J.flatten() - JP_t1_t.flatten())**2),
                            np.mean((J.flatten() - JP_t.flatten())**2),
                            np.mean((J.flatten() - JP_t1.flatten())**2),
                            np.mean((J.flatten() - JP2_t.flatten())**2)])


cmap = cm.get_cmap('plasma_r')
colors = []
for i in range(4):
    colors += [cmap((i) / 3)]
line = [(5, 4), (5, 4), '', '']

labels = [
    r'Plefka[$t-1,t$]',
    r'Plefka[$t$]',
    r'Plefka[$t-1$]',
    r'Plefka2[$t$]',
    r'Original']
lws = [2, 2, 2, 3, 1.5]


filename = filename = 'data/inverse/inverse_100_R_' + str(R) + '.npz'
data = np.load(filename)
HP_t1_t = data['HP_t1_t']
HP_t = data['HP_t']
HP_t1 = data['HP_t1']
HP2_t = data['HP2_t']
JP_t1_t = data['JP_t1_t']
JP_t = data['JP_t']
JP_t1 = data['JP_t1']
JP2_t = data['JP2_t']
H = data['H']
J = data['J']


letters = ['A', 'B', 'C', 'D']
pos_l = [-0.2 * 2 / 3, 1.0]
fig, ax = plt.subplots(2, 2, figsize=(16 * 2 / 3, 10 * 2 / 3), dpi=300)
ax[0, 0].plot(H, HP_t1_t, 'v', color=colors[0], ms=5, rasterized=True)
ax[0, 0].plot(H, HP_t, 's', color=colors[1], ms=5, rasterized=True)
ax[0, 0].plot(H, HP_t1, 'd', color=colors[2], ms=5, rasterized=True)
ax[0, 0].plot(H, HP2_t, 'o', color=colors[3], ms=5, rasterized=True)
ax[0, 0].plot([np.min(H), np.max(H)], [np.min(H), np.max(H)], 'k', lw=2.5)
ax[0, 0].axis([np.min(H), np.max(H), -3.5, 1])
ax[0, 0].set_xlabel(r'$H_i^o$', fontsize=18)
ax[0, 0].set_ylabel(r'$H_i^p$', fontsize=18, rotation=0, labelpad=15)
ax[0, 0].text(pos_l[0], pos_l[1], r'\textbf ' + letters[0],
              transform=ax[0, 0].transAxes, fontsize=20, va='top', ha='right')

ax[0, 1].plot(J.flatten(), JP_t1_t.flatten(), 'v',
              color=colors[0], ms=5, rasterized=True)
ax[0, 1].plot(J.flatten(), JP_t.flatten(), 's',
              color=colors[1], ms=5, rasterized=True)
ax[0, 1].plot(J.flatten(), JP_t1.flatten(), 'd',
              color=colors[2], ms=5, rasterized=True)
ax[0, 1].plot(J.flatten(), JP2_t.flatten(), 'o',
              color=colors[3], ms=5, rasterized=True)
ax[0, 1].plot([np.min(J), np.max(J)], [np.min(J), np.max(J)], 'k', lw=2.5)
ax[0, 1].set_xlabel(r'$J_{ij}^o$', fontsize=18)
ax[0, 1].set_ylabel(r'$J_{ij}^p$', fontsize=18, rotation=0, labelpad=15)
ax[0, 1].text(pos_l[0], pos_l[1], r'\textbf ' + letters[1],
              transform=ax[0, 1].transAxes, fontsize=20, va='top', ha='right')
ax[0, 1].axis([np.min(J), np.max(J), np.min(J), np.max(J) * 2])

for m in range(M):
    ax[1, 0].semilogy(betas, errorH[m, :], dashes=line[m],
                      color=colors[m], lw=lws[m])  # , label=labels[m])
for m in range(2):
    ax[1, 0].semilogy(betas, errorH[m, :], dashes=line[m],
                      color=colors[m], lw=lws[m])  # , label=labels[m])
ax[1, 0].set_xlabel(r'$\beta/\beta_c$', fontsize=18)
ax[1, 0].set_ylabel(r'$\epsilon_H$', fontsize=18, rotation=0, labelpad=15)
ax[1, 0].text(pos_l[0], pos_l[1], r'\textbf ' + letters[2],
              transform=ax[1, 0].transAxes, fontsize=20, va='top', ha='right')
for m in range(M):
    ax[1, 1].semilogy(betas, errorJ[m, :], dashes=line[m],
                      color=colors[m], lw=lws[m], label=labels[m])
for m in range(2):
    ax[1, 1].semilogy(betas, errorJ[m, :], dashes=line[m],
                      color=colors[m], lw=lws[m])
ax[1, 1].set_xlabel(r'$\beta/\beta_c$', fontsize=18)
ax[1, 1].set_ylabel(r'$\epsilon_J$', fontsize=18, rotation=0, labelpad=15)

ax[1, 1].text(pos_l[0], pos_l[1], r'\textbf ' + letters[3],
              transform=ax[1, 1].transAxes, fontsize=20, va='top', ha='right')

plt.figlegend(
    loc='upper center',
    bbox_to_anchor=(
        0.5,
        1.),
    borderaxespad=0,
    ncol=5)
fig.tight_layout(h_pad=0.3, w_pad=0.7, rect=[0, 0, 1, 0.975])

plt.savefig('img/results-inverse-Ising.pdf', bbox_inches='tight')
