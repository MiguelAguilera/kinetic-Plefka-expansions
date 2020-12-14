#!/usr/bin/env python3
"""
GPLv3 2020 Miguel Aguilera

This code displays the results of the forward Ising problem computed
from running "generate_data.py" and "forward-Ising-problem.py"
"""

import context
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib


plt.rc('text', usetex=True)
font = {'size': 15}
plt.rc('font', **font)
plt.rc('legend', **{'fontsize': 18})


size = 512
beta = 1.0
R = 1000000
mode = 'c'
gamma1 = 0.5
gamma2 = 0.1


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

pos_l = [-0.2, 1.0]

letters = ['A', 'B', 'C', 'D', 'E', 'F']


filename = 'data/forward/forward_100_R_' + str(R) + '.npz'
data = np.load(filename)

time_P_t1_t = data['time_P_t1_t']
time_P_t = data['time_P_t']
time_P_t1 = data['time_P_t1']
time_P2_t = data['time_P2_t']


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

plt.figure()
plt.plot(EDP_t1_t)
plt.plot(EDP_t)
plt.plot(EDP_t1)
plt.plot(EDP2_t)

fig, ax = plt.subplots(1, 3, figsize=(16, 1 + 9 * 1 / 3), dpi=300)
ax[0].loglog(time_P_t1_t, np.mean(EmP_t1_t), 'v', ms=12, color=colors[0])
ax[0].loglog(time_P_t, np.mean(EmP_t), 's', ms=12, color=colors[1])
ax[0].loglog(time_P_t1, np.mean(EmP_t1), 'd', ms=12, color=colors[2])
ax[0].loglog(time_P2_t, np.mean(EmP2_t), 'o', ms=12, color=colors[3])
ax[0].set_xticks([], minor=True)
ax[0].set_xticks([1, 10, 100, 1000])
ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[0].set_xlabel(r'execution time (seconds)', fontsize=18)
ax[0].set_ylabel(r'$\epsilon_m$', fontsize=18, rotation=0, labelpad=15)
ax[0].text(
    pos_l[0],
    pos_l[1],
    r'\textbf ' +
    letters[0],
    transform=ax[0].transAxes,
    fontsize=20,
    va='top',
    ha='right')

ax[1].loglog(time_P_t1_t, np.mean(ECP_t1_t), 'v', ms=12, color=colors[0])
ax[1].loglog(time_P_t, np.mean(ECP_t), 's', ms=12, color=colors[1])
ax[1].loglog(time_P_t1, np.mean(ECP_t1), 'd', ms=12, color=colors[2])
ax[1].loglog(time_P2_t, np.mean(ECP2_t), 'o', ms=12, color=colors[3])
ax[1].set_xticks([], minor=True)
ax[1].set_xticks([1, 10, 100, 1000])
ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[1].set_xlabel(r'execution time (seconds)', fontsize=18)
ax[1].set_ylabel(r'$\epsilon_C$', fontsize=18, rotation=0, labelpad=15)
ax[1].text(
    pos_l[0],
    pos_l[1],
    r'\textbf ' +
    letters[1],
    transform=ax[1].transAxes,
    fontsize=20,
    va='top',
    ha='right')

ax[2].loglog(
    time_P_t1_t,
    np.mean(EDP_t1_t),
    'v',
    ms=12,
    color=colors[0],
    label=labels[0])
ax[2].loglog(
    time_P_t,
    np.mean(EDP_t),
    's',
    ms=12,
    color=colors[1],
    label=labels[1])
ax[2].loglog(
    time_P_t1,
    np.mean(EDP_t1),
    'd',
    ms=12,
    color=colors[2],
    label=labels[2])
ax[2].loglog(
    time_P2_t,
    np.mean(EDP2_t),
    'o',
    ms=12,
    color=colors[3],
    label=labels[3])
ax[2].set_xticks([], minor=True)
ax[2].set_xticks([1, 10, 100, 1000])
ax[2].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[2].set_xlabel(r'execution time (seconds)', fontsize=18)
ax[2].set_ylabel(r'$\epsilon_D$', fontsize=18, rotation=0, labelpad=15)
ax[2].text(
    pos_l[0],
    pos_l[1],
    r'\textbf ' +
    letters[2],
    transform=ax[2].transAxes,
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
plt.savefig('img/results-execution_time-forward.pdf', bbox_inches='tight')
# plt.show()


filename = 'data/inverse/inverse_100_R_' + str(R) + '.npz'
data = np.load(filename)

time_P_t1_t = data['time_P_t1_t']
time_P_t = data['time_P_t']
time_P_t1 = data['time_P_t1']
time_P2_t = data['time_P2_t']

H = data['H']
J = data['J']

EHP_t1_t = np.mean((H - data['HP_t1_t'])**2)
EHP_t = np.mean((H - data['HP_t'])**2)
EHP_t1 = np.mean((H - data['HP_t1'])**2)
EHP2_t = np.mean((H - data['HP2_t'])**2)
EJP_t1_t = np.mean((J - data['JP_t1_t'])**2)
EJP_t = np.mean((J - data['JP_t'])**2)
EJP_t1 = np.mean((J - data['JP_t1'])**2)
EJP2_t = np.mean((J - data['JP2_t'])**2)


fig, ax = plt.subplots(1, 2, figsize=(16 * 2 / 3, 1 + 9 * 1 / 3), dpi=300)
ax[0].loglog(time_P_t1_t, np.mean(EHP_t1_t), 'v', ms=12, color=colors[0])
ax[0].loglog(time_P_t, np.mean(EHP_t), 's', ms=12, color=colors[1])
ax[0].loglog(time_P_t1, np.mean(EHP_t1), 'd', ms=12, color=colors[2])
ax[0].loglog(time_P2_t, np.mean(EHP2_t), 'o', ms=12, color=colors[3])
ax[0].set_xticks([], minor=True)
ax[0].set_xticks([10, 100, 1000])
ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[0].set_xlabel(r'execution time (seconds)', fontsize=18)
ax[0].set_ylabel(r'$\epsilon_H$', fontsize=18, rotation=0, labelpad=15)
ax[0].text(
    pos_l[0],
    pos_l[1],
    r'\textbf ' +
    letters[0],
    transform=ax[0].transAxes,
    fontsize=20,
    va='top',
    ha='right')

ax[1].loglog(
    time_P_t1_t,
    np.mean(EJP_t1_t),
    'v',
    ms=12,
    color=colors[0],
    label=labels[0])
ax[1].loglog(
    time_P_t,
    np.mean(EJP_t),
    's',
    ms=12,
    color=colors[1],
    label=labels[1])
ax[1].loglog(
    time_P_t1,
    np.mean(EJP_t1),
    'd',
    ms=12,
    color=colors[2],
    label=labels[2])
ax[1].loglog(
    time_P2_t,
    np.mean(EJP2_t),
    'o',
    ms=12,
    color=colors[3],
    label=labels[3])
ax[1].set_xticks([], minor=True)
ax[1].set_xticks([10, 100, 1000])
ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[1].set_xlabel(r'execution time (seconds)', fontsize=18)
ax[1].set_ylabel(r'$\epsilon_J$', fontsize=18, rotation=0, labelpad=15)
ax[1].text(
    pos_l[0],
    pos_l[1],
    r'\textbf ' +
    letters[1],
    transform=ax[1].transAxes,
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
plt.savefig('img/results-execution_time-inverse.pdf', bbox_inches='tight')
