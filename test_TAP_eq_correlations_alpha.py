from mf_ising import mf_ising
#from plefka_ising import plefka_ising
from kinetic_ising import ising
import numpy as np
from matplotlib import pyplot as plt

size = 10
# size=5
I = ising(size)
#
beta = 1.
gamma1 = 0.3
gamma2 = 0.3
I.h = 0.0 + gamma1 * np.random.randn(size)
I.J = beta * (1 + gamma2 * np.random.randn(size, size)) / size
#I.J = 1.0 * np.random.randn(size, size) / size * 1

R = 400
T=32
N = 100
m = np.zeros((size, N))
C = np.zeros((size, size, N))
m_exp = np.zeros(size)
C_exp = np.zeros((size, size))
#for rep in range(R // 5):
#    I.GlauberStep()

for rep in range(R):
    I.randomize_state()
    for t in range(T):
        I.GlauberStep()
    h = I.h + np.dot(I.J, I.s)
    m_exp += np.tanh(h) / R
#	m_prev += I.s/R
    C_exp += np.einsum('i,j->ij', np.tanh(h), np.tanh(h), optimize=True) / R
C_exp -= np.einsum('i,j->ij', m_exp, m_exp, optimize=True)
print(m_exp)
print(C_exp)
H = I.h + np.dot(I.J, m_exp)
alpha = np.linspace(0, 1, N)
V = np.einsum('ij,kl,jl->ik', I.J, I.J, C_exp, optimize=True)
Lc = np.sqrt(np.diag(V))
G01 = V[0, 1] / Lc[0] / Lc[1]

#J1= np.zeros((size,size))
# J1=np.ones((size,size))*np.mean(I.J)
# J1=np.zeros((size,size))
# for i in range(size):
#	J1[i,:]=np.ones(size)*np.mean(I.J[i,:])
print(G01)

H = I.H + np.dot(I.J, m_exp)

Vii = np.einsum('ij,il,jl->i', I.J, I.J, C_exp, optimize=True)

for rep in range(R):
    
    I.randomize_state()
    for t in range(T):
        I.GlauberStep()
    dh = np.dot(I.J, I.s - m_exp)
#	sigma0=np.sign(dh[0])
#	sigma1=np.sign(dh[1])
    h = np.zeros((size, N))
    for i in range(N):
#		h[:,i] = H[0:2] + (1-alpha[i])*sigma*Lc[0:2] + alpha[i]*dh[0:2]
        h[:, i] = H + alpha[i] * dh + (1 - alpha[i]) * m_exp * Vii
#		h[:,i] = H[0:2] + (1-alpha[i])*x + alpha[i]*dh[0:2]
    m += np.tanh(h) / R
    C += np.einsum('in,jn->ijn', np.tanh(h), np.tanh(h), optimize=True) / R
#C -= np.einsum('in,jn->ijn',m,m, optimize=True)

plt.figure()
plt.plot(m[0, :])
plt.plot(m[1, :])
plt.figure()
plt.plot(alpha, C[0, 1, :] - m[0, :] * m[1, :])
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$C$')
# plt.figure()
# plt.plot(alpha[0:-1],np.diff(C[0,1,:]))
# plt.figure()
# plt.plot(alpha[0:-2],np.diff(np.diff(C[0,1,:])))
#
d1=np.diff(C[0,1,:])[0]/(alpha[1]-alpha[0])
d2=np.diff(np.diff(C[0,1,:]))[0]/(alpha[1]-alpha[0])**2
# plt.figure()
plt.plot(alpha,alpha[0]+d1*alpha+0.5*d2*alpha**2)

plt.show()
