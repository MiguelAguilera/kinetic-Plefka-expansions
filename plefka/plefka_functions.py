#!/usr/bin/env python3
"""
GPLv3 2020 Miguel Aguilera

This code defines functions for applying Plefka expansions for mean field simulations
"""
import numpy as np

def TAP_eq(x, H, Vii):
    return np.tanh(H - x * Vii) - x


def diff_TAP_eq(x, H, Vii):
    return -Vii * (1 - np.tanh(H - x * Vii)**2) - 1


def solve_TAP_eq(x0, H, Vii, TOL=1E-15):
    x = x0.copy()
    TAP = TAP_eq(x, H, Vii)
    error = np.max(np.abs(TAP))
    count = 0
    while error > TOL:
        TAP = TAP_eq(x, H, Vii)
        dTAP = diff_TAP_eq(x, H, Vii)
        x -= TAP / dTAP * (np.abs(TAP) > TOL).astype(int)
        error = np.max(np.abs(TAP))
    return x


# PLEFKA[t-1,t], order 1


def update_m_P_t1_t_o1(H, J, m):
    return np.tanh(H + np.dot(J, m))


def update_C_P_t1_t_o1(H, J, m):
    return np.diag(1 - m**2)


def update_D_P_t1_t_o1(H, J, m, m_p):
    D = np.einsum('i,il,l->il', 1 - m**2, J, 1 - m_p**2, optimize=True)
    return D

# PLEFKA[t-1,t], order 2


def update_m_P_t1_t_o2(H, J, m_p):
    Vii = np.einsum('ij,j->i', J**2, 1 - m_p**2, optimize=True)
    Heff = H + np.dot(J, m_p)
    return solve_TAP_eq(m_p.copy(), Heff, Vii)


def update_C_P_t1_t_o2(H, J, m, m_p):
    C = np.einsum(
        'i,k,ij,kj,j->ik',
        1 - m**2,
        1 - m**2,
        J,
        J,
        1 - m_p**2,
        optimize=True)
    np.einsum('ii->i', C, optimize=True)[:] = 1 - m**2
    return C


def update_D_P_t1_t_o2(H, J, m, m_p):
    D = np.einsum('i,il,l->il', 1 - m**2, J, 1 - m_p**2, optimize=True)
    D *= 1 + 2*np.einsum('i,il,l->il', m, J, m_p, optimize=True)
    return D

# PLEFKA[t], order 1


def update_m_P_t_o1(H, J, m):
    return np.tanh(H + np.dot(J, m))


def update_C_P_t_o1(H, J, m_p):
    return np.diag(1 - m_p**2)


def update_D_P_t_o1(H, J, m, C_p):
    D = np.einsum('i,ij,jl->il', 1 - m**2, J, C_p, optimize=True)
    return D

# PLEFKA[t], order 2


def update_m_P_t_o2(H, J, m_p, C_p):
    Vii = np.einsum('ij,il,jl->i', J, J, C_p, optimize=True)
    Heff = H + np.dot(J, m_p)
    return solve_TAP_eq(m_p.copy(), Heff, Vii)


def update_C_P_t_o2(H, J, m, C_p):
    C = np.einsum(
        'i,k,ij,kl,jl->ik',
        1 - m**2,
        1 - m**2,
        J,
        J,
        C_p,
        optimize=True)
    np.einsum('ii->i', C, optimize=True)[:] = 1 - m**2
    return C


def update_D_P_t_o2(H, J, m, m_p, C_p):
    D = np.einsum('i,ij,jl->il', 1 - m**2, J, C_p, optimize=True)
    D += np.einsum('i,ij, il ,jl,l->il', 2 * m * (1 - m**2),
                   J, J, C_p, m_p, optimize=True)
    return D


# PLEFKA[t-1], order 1
def integrate_1DGaussian(f, args=(), Nint=20):
    x = np.linspace(-1, 1, Nint) * 4
    return np.sum(f(x, *args)) * (x[1] - x[0])


def integrate_2DGaussian(f, args=(), Nx=20, Ny=20):
    p = np.linspace(-1, 1, Nx) * 4
    n = np.linspace(-1, 1, Ny) * 4
    P, N = np.meshgrid(p, n)
    return np.sum(f(P, N, *args)) * (p[1] - p[0]) * (n[1] - n[0])


def dT1(x, g, D):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2) * \
        np.tanh(g + x * np.sqrt(D))


def dT1_1(x, g, D):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2) * \
        (1 - np.tanh(g + x * np.sqrt(D))**2)


def dT1_2(x, g, D):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2) * (-2 *
                                                           np.tanh(g + x * np.sqrt(D))) * (1 - np.tanh(g + x * np.sqrt(D))**2)


def update_m_P_t1_o1(H, J, m_p):
    size = len(H)
    m = np.zeros(size)
    g = H + np.dot(J, m_p)
    D = np.dot(J**2, 1 - m_p**2)
    for i in range(size):
        m[i] = integrate_1DGaussian(dT1, (g[i], D[i]))
    return m


def update_D_P_t1_o1(H, J, m_p, C_p):
    size = len(H)
    a = np.zeros(size)
    g = H + np.dot(J, m_p)
    D = np.dot(J**2, 1 - m_p**2)
    for i in range(size):
        a[i] = integrate_1DGaussian(dT1_1, (g[i], D[i]))
    return np.einsum('i,ij,jl->il', a, J, C_p)


def dT2_rot(p, n, gx, gy, Dx, Dy, rho):
    if n is None:
        return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * p**2) * np.tanh(gx + p * np.sqrt(
            1 + rho) * np.sqrt(Dx / 2)) * np.tanh(gy + p * np.sqrt(1 + rho) * np.sqrt(Dy / 2))
    elif p is None:
        return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * p**2) * np.tanh(gx + n * np.sqrt(
            1 - rho) * np.sqrt(Dx / 2)) * np.tanh(gy - n * np.sqrt(1 - rho) * np.sqrt(Dy / 2))
    else:
        return 1 / (2 * np.pi) * np.exp(-0.5 * (p**2 + n**2))  \
            * np.tanh(gx + (p * np.sqrt(1 + rho) + n * np.sqrt(1 - rho)) * np.sqrt(Dx / 2)) \
            * np.tanh(gy + (p * np.sqrt(1 + rho) - n * np.sqrt(1 - rho)) * np.sqrt(Dy / 2))


def update_C_P_t1_o1(H, J, m, m_p, C_p):
    size = len(H)
    C = np.zeros((size, size))
    g = H + np.dot(J, m_p)
    D = np.dot(J**2, 1 - m_p**2)
    inv_D = np.zeros(size)
    inv_D[D > 0] = 1 / D[D > 0]
    rho = np.einsum('i,k,ij,kj,j->ik', np.sqrt(inv_D),
                    np.sqrt(inv_D), J, J, 1 - m_p**2, optimize=True)
    for i in range(size):
        C[i, i] = 1 - m[i]**2
        for j in range(i + 1, size):
            if rho[i, j] > (1 - 1E5):
                C[i, j] = integrate_1DGaussian(
                    dT2_rot, (None, g[i], g[j], D[i], D[j], rho[i, j])) - m[i] * m[j]
            else:
                C[i, j] = integrate_2DGaussian(
                    dT2_rot, (g[i], g[j], D[i], D[j], rho[i, j])) - m[i] * m[j]
            C[j, i] = C[i, j]
    return C

# PLEFKA2[t], order 2

def TAP_eq_D(x, Heff, V):
    return x - Heff + np.tanh(x) * V


def diff_TAP_eq_D(x, V):
    return 1 + (1 - np.tanh(x)**2) * V


def solve_TAP_eq_D(x0, Heff, V, TOL=1E-15):
    x = x0.copy()
    TAP = TAP_eq_D(x, Heff, V)
    error = np.max(np.abs(TAP))
    while error > TOL:
        TAP = TAP_eq_D(x, Heff, V)
        dTAP = diff_TAP_eq_D(x, V)
        x -= TAP / dTAP * (np.abs(TAP) > TOL).astype(int)
        error = np.max(np.abs(TAP))
    return x


def update_D_P2_t_o2(H, J, m_p, C_p, D_p):
    size = len(H)
    D = np.zeros((size, size))
    m_D = np.zeros(size)

    Heff = H + np.dot(J, m_p)
    Heff_i = np.einsum('i,il->il', Heff, np.ones((size, size)))
    V_p = np.einsum('ij,in,jn->i', J, J, C_p, optimize=True)
    W_p = np.einsum('ij,ln,jn->il', J, J, D_p, optimize=True)

    m_i = np.zeros((size, size))
    m_pil = np.einsum('l,il->il', m_p, np.ones((size, size)))
    V_pil = np.einsum('i,il->il', V_p, np.ones((size, size)))
    V_pil -= 2 * np.einsum('il,in,ln->il', J, J, C_p)
    V_pil += np.einsum('il,ll->il', J**2, C_p)
    W_pil = W_p - np.einsum('il,ln,ln->il', J, J, D_p, optimize=True)
    
    Delta_il = J + W_pil

    for sl in [-1, 1]:
        Heff_il = Heff_i + Delta_il * (sl - m_pil)
        theta = solve_TAP_eq_D(Heff_il, Heff_il, V_pil)
        D += np.tanh(theta) * sl * (1 + sl * m_pil) / 2
        m_i += np.tanh(theta) * (1 + sl * m_pil) / 2

    D -= m_i * m_pil
    m_D = np.einsum('il->i', m_i / size, optimize=True)
    return m_D, D



def update_C_P2_t_o2(H, J, m, m_p, C_p):
    size = len(H)
    C_D = np.zeros((size, size))

    Heff = H + np.dot(J, m_p)
    V_p = np.einsum('ij,il,jl->i', J, J, C_p, optimize=True)
    V_pik = np.einsum('i,ik->ik', V_p, np.ones((size, size)))
    W_p = np.einsum('ij,kl,jl->ik', J, J, C_p, optimize=True)

    m_i = np.zeros((size, size))
    m_pik = np.einsum('k,ik->ik', m_p, np.ones((size, size)))
    Heff_i = np.einsum('i,il->il', Heff, np.ones((size, size)))
    Delta_ik = W_p

    for sk in [-1, 1]:
        Heff_ik = Heff_i + Delta_ik * (sk - m_pik)
        theta = solve_TAP_eq_D(Heff_ik, Heff_ik, V_pik)
        C_D += np.tanh(theta) * (sk - m_pik) * (1 + sk * m_pik) / 2

    C_D = 0.5 * (C_D + C_D.T)
    np.einsum('ii->i', C_D)[:] = 1 - m**2
    return C_D

# PLEFKA_C (Old code from https://arxiv.org/abs/2002.04309v1)
# def correlation_Ising_2nodes(Heff_i, Heff_j, Jeff):
#    size=Heff_i.shape[0]
#    s1 = np.array([-1, 1])
#    s2 = np.array([-1, 1])
#    S1 = np.einsum('i,j->ij', s1, np.ones(2), optimize=True)
#    S2 = np.einsum('i,j->ij', np.ones(2), s2, optimize=True)
#    P=np.zeros((2,2,size,size))
#    S=np.array([-1, 1])
#    for ind1, s1 in enumerate(S):
#        for ind2, s2 in enumerate(S):
#            P[ind1,ind2,:,:] = np.exp(s1*Heff_i + s2*Heff_j + s1*s2*Jeff)
#    Z = np.einsum('abij->ij',P)
#    P = np.einsum('abij,ij->abij',P,1/Z)
#    m1 = np.einsum('abij,a->ij',P,S)
#    m2 = np.einsum('abij,b->ij',P,S)
#    C12 = np.einsum('abij,a,b->ij',P,S,S) - m1 * m2
#    m12 = (np.einsum('ij->i',m1) + np.einsum('ij->j',m2)) * 0.5 / size
#    return m12,C12


# def update_P_t1_t_o2(H, J, m, m_p, C_p):
#    size = len(H)
#    V_p = np.einsum('ij,kl,jl->ik', J, J, C_p, optimize=True)
#    Heff = H + np.dot(J, m_p)
#    Heff_i = np.einsum('i,ij->ij',Heff - m*np.diag(V_p),np.ones((size,size)), optimize=True)
#    Heff_i -= np.einsum('j,ij->ij',m,V_p, optimize=True)
#    Heff_j = np.einsum('j,ij->ij',Heff - m*np.diag(V_p),np.ones((size,size)), optimize=True)
#    Heff_j -= np.einsum('i,ij->ij',m,V_p, optimize=True)
#
#    Jeff = V_p
#    m_C, C = correlation_Ising_2nodes(Heff_i,Heff_j, Jeff)
#    np.einsum('ii->i', C, optimize=True)[:] = 1 - m**2
#    return m_C,C


# def correlation_Ising_2nodes__(H1, H2, J):
#    s1 = np.array([-1, 1])
#    s2 = np.array([-1, 1])
#    S1 = np.einsum('i,j->ij', s1, np.ones(2), optimize=True)
#    S2 = np.einsum('i,j->ij', np.ones(2), s2, optimize=True)
#    P = np.exp(S1 * H1 + S2 * H2 + S1 * S2 * J)
#    P /= np.sum(P)
#    m1 = np.sum(S1 * P)
#    m2 = np.sum(S2 * P)
#    C12 = np.sum(S1 * S2 * P) - m1 * m2
#    return m1, m2, C12
#
# def update_P_t1_t_o2__(H, J, m, m_p, V_p):
#    size = len(H)
#    m_C = np.zeros(size)
#    C = np.zeros((size, size))
#    Heff = H + np.dot(J, m_p)
#    for i in range(size):
#        C[i, i] = 1 - m[i]**2
#        for j in range(i + 1, size):
#            Heff_i=Heff[i] - m[i]*V_p[i,i] - m[j]*V_p[i,j]
#            Heff_j=Heff[j] - m[j]*V_p[j,j] - m[i]*V_p[i,j]
#            mi, mj, Cij = correlation_Ising_2nodes__(Heff_i, Heff_j, V_p[i, j])
#            C[i, j] = Cij# - m[i]*m[j]
#            C[j, i] = Cij# - m[i]*m[j]
#            m_C[i] += mi / (size - 1)
#            m_C[j] += mj / (size - 1)
#    return m_C,C
