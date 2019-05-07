#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:32:23 2019

@author: maguilera
"""
import numpy as np
import scipy.optimize


def TAP_eq(x, H, Vii):
    return np.tanh(H - x * Vii) - x
#	return np.tanh(H - 1/x + x*sC) - x
#	return np.tanh(H - (1-np.sqrt(1-4*x**2*sC))/2/x) - x


def update_m_P0_o1(H, J, m):
    return np.tanh(H + np.dot(J, m))


def update_m_P0_o2(H, J, m):
    Vii = np.einsum('ij,j->i', J**2, 1 - m**2, optimize=True)
    h = H + np.dot(J, m)
    return scipy.optimize.fsolve(TAP_eq, m.copy(), args=(h, Vii))


def update_C_P0_o1(H, J, m):
    return np.diag(1 - m**2)


def update_C_P0_o2(H, J, m, m_p):
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


def update_m_P0_o2_(H, J, m):
    Vii = np.einsum('ij,j->i', J**2, 1 - m**2, optimize=True)
    h = H + np.dot(J, m)
    Vii += (h - np.arctanh(m))**2
    return scipy.optimize.fsolve(TAP_eq, m.copy(), args=(h, Vii))


def update_C_P0_o2_(H, J, m, m_p):
    C = np.einsum(
        'i,k,ij,kj,j->ik',
        1 - m**2,
        1 - m**2,
        J,
        J,
        1 - m_p**2,
        optimize=True)
    dh = H + np.dot(J, m) - np.arctanh(m)
    C += np.einsum('i,k->ik', (1 - m**2) * dh, (1 - m**2) * dh, optimize=True)
    np.einsum('ii->i', C, optimize=True)[:] = 1 - m**2
    return C

def update_D_P0_o1(H, J, m, m_p):
    D = np.einsum('i,il,l->il', 1-m**2, J, 1-m_p**2, optimize=True)
    return D
    
def update_D_P0_o2(H, J, m, m_p):
    D = np.einsum('i,il,l->il', 1-m**2, J, 1-m_p**2, optimize=True)
    D *= 1 + np.einsum('i,il,l->il', m, J, m_p, optimize=True)
    return D
    
def update_m_P1_o1(H, J, m):
    return np.tanh(H + np.dot(J, m))


def update_m_P1_o2(H, J, m, C):
    Vii = np.einsum('ij,il,jl->i', J, J, C, optimize=True)
    h = H + np.dot(J, m)
    return scipy.optimize.fsolve(TAP_eq, m.copy(), args=(h, Vii))


def update_C_P1_o1(H, J, m):
    return np.diag(1 - m**2)

def update_C_P1_o2(H, J, m, C_p):
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

def update_D_P1_o1(H, J, m, C_p):
    D = np.einsum('i,ij,jl->il', 1-m**2, J, C_p, optimize=True)
    return D
    
def update_D_P1_o2(H, J, m, m_p, C_p):
    D = np.einsum('i,ij,jl->il', 1-m**2, J, C_p, optimize=True)
    D -= np.einsum('i,ij,jl,j->il', 2*m*(1-m**2), J**2, C_p, m_p, optimize=True)
    D -= np.einsum('i,ij, il ,jl,l->il', 4*m*(1-m**2), J, J, C_p, m_p, optimize=True)
    return D

def update_m_P1_o2_(H, J, m, C):
    Vii = np.einsum('ij,il,jl->i', J, J, C, optimize=True)
    h = H + np.dot(J, m)
    Vii += (h - np.arctanh(m))**2
    return scipy.optimize.fsolve(TAP_eq, m.copy(), args=(h, Vii))

def update_C_P1_o2_(H, J, m, C_p):
    C = np.einsum(
        'i,k,ij,kl,jl->ik',
        1 - m**2,
        1 - m**2,
        J,
        J,
        C_p,
        optimize=True)
    dh = H + np.dot(J, m) - np.arctanh(m)
    C += np.einsum('i,k->ik', (1 - m**2) * dh, (1 - m**2) * dh, optimize=True)
    np.einsum('ii->i', C, optimize=True)[:] = 1 - m**2
    return C


def correlation_Ising_2nodes(H1, H2, J):
    s1 = np.array([-1, 1])
    s2 = np.array([-1, 1])
    S1 = np.einsum('i,j->ij', s1, np.ones(2), optimize=True)
    S2 = np.einsum('i,j->ij', np.ones(2), s2, optimize=True)
    P = np.exp(S1 * H1 + S2 * H2 + S1 * S2 * J)
    P /= np.sum(P)
    m1 = np.sum(S1 * P)
    m2 = np.sum(S2 * P)
    C12 = np.sum(S1 * S2 * P) - m1 * m2
    return m1, m2, C12


def update_P1C_o2(H, J, m, m_p, V_p):
    size = len(H)
#    m = np.zeros(size)
    C = np.zeros((size, size))
    Heff = H + np.dot(J, m_p)
    for i in range(size):
        C[i, i] = 1 - m[i]**2
        for j in range(i + 1, size):
            m1, m2, C12 = correlation_Ising_2nodes(Heff[i], Heff[j], V_p[i, j])
            C[i, j] = C12
            C[j, i] = C12
#            m[i] += m1 / (size - 1)
#            m[j] += m2 / (size - 1)
    return C

def update_P1D_o1(H, J, m_p):
    size = len(H)
    D = np.zeros((size, size))
    m = np.zeros(size)
    Heff = H + np.dot(J, m_p)
    for i in range(size):
        for j in range(size):
            m_ij = 0
            for sj in [-1,1]:
                Theta = Heff[i] - J[i,j]*m_p[j]
                D[i,j] += np.tanh(Theta+J[i,j]*sj)*sj*(1+sj*m_p[j])/2
                m_ij += np.tanh(Theta+J[i,j]*sj)*sj*(1+sj*m_p[j])/2
            D[i,j] -= m_ij*m_p[j]
    return D

def TAP_eq_D(x, Heff_i, Jijsj, V_pij):
    return x-Heff_i + np.tanh(x + Jijsj)*V_pij

def update_P1D_o2(H, J, m_p, C_p):
    size = len(H)
    D = np.zeros((size, size))
    m1 = np.zeros(size)
    Heff = H + np.dot(J, m_p)
    V_p = np.einsum('ij,il,jl->i',J,J,C_p,optimize=True)
    for i in range(size):
        for j in range(size):
            m_ij = 0
            for sj in [-1,1]:
                Heff_i = Heff[i] - J[i,j]*m_p[j]
                V_pij = V_p[i] - 2  * J[i,j] * np.dot(J[i,:], C_p[:,j]) + J[i,j]**2 * C_p[j,j]
                Theta = scipy.optimize.fsolve(TAP_eq_D, Heff_i, args=(Heff_i, J[i,j]*sj, V_pij))
                D[i,j] += np.tanh(Theta+J[i,j]*sj)*sj*(1+sj*m_p[j])/2
                m_ij += np.tanh(Theta+J[i,j]*sj)*(1+sj*m_p[j])/2
            D[i,j] -= m_ij*m_p[j]
    return D

def integrate_1DGaussian(f, args=(), Nint=50):
    x = np.linspace(-1, 1, Nint) * 4
    return np.sum(f(x, *args)) * (x[1] - x[0])


def dT1(x, g, D):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2) * \
        np.tanh(g + x * np.sqrt(D))


def dT1_1(x, g, D):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2) * \
        (1 - np.tanh(g + x * np.sqrt(D))**2)


def dT1_2(x, g, D):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2) * (-2 *
                                                           np.tanh(g + x * np.sqrt(D))) * (1 - np.tanh(g + x * np.sqrt(D))**2)


def update_m_P2_o1(H, J, m_p):
    size = len(H)
    m = np.zeros(size)
    g = H + np.dot(J, m_p)
    D = np.dot(J**2, 1 - m_p**2)
    for i in range(size):
        m[i] = integrate_1DGaussian(dT1, (g[i], D[i]))
    return m


def update_D_P2_o1(H, J, m_p):
    size = len(H)
    m = np.zeros(size)
    D = np.zeros((size,size))
    g = H + np.dot(J, m_p)
    Dlt = np.dot(J**2, 1 - m_p**2)
    for i in range(size):
        m[i] = integrate_1DGaussian(dT1, (g[i], Dlt[i]))
        intT1_1 = integrate_1DGaussian(dT1_1, (g[i], Dlt[i]))
        intT1_2 = integrate_1DGaussian(dT1_2, (g[i], Dlt[i]))
        for j in range(size):
             D[i,j] += (1-m_p[j]**2)*J[i,j] * intT1_1
             D[i,j] += (1-m_p[j]**2)*J[i,j]**2 * m_p[j] * intT1_2
#             D[i,j] -= m[i]*m_p[j]
    return m, D
#    
#    
#def update_m_P2_o2(H, J, m_p, V_pp):
#    size = len(H)
#    m = np.zeros(size)
#    g = H + np.dot(J, m_p)
#    D = np.dot(J**2, 1 - m_p**2)
#    for i in range(size):
#        m[i] = integrate_1DGaussian(dT1, (g[i], D[i]))
##		m[i] -= np.sum(m_p*J[i,:]*(1-m_p**2)*np.diag(V_pp))*integrate_1DGaussian(dT1_1,(g[i], D[i]))
#        m[i] += 0.5 * np.einsum('k,l,kl->',
#                                (1 - m_p**2) * J[i,
#                                                 :],
#                                (1 - m_p**2) * J[i,
#                                                 :],
#                                V_pp) * integrate_1DGaussian(dT1_2,
#                                                             (g[i],
#                                                              D[i]))
#    return m


#def dT2(x, y, gx, gy, Dx, Dy, rho):
#    return 1 / (2 * np.pi * np.sqrt(1 - rho**2)) * np.exp(-1 / (1 - rho**2) * (0.5 * x**2 + \
#                0.5 * y**2 - rho * x * y)) * np.tanh(gx + x * np.sqrt(Dx)) * np.tanh(gy + y * np.sqrt(Dy))


#def dT2_1(x, y, gx, gy, Dx, Dy, rho, my):
#    return 1 / (2 * np.pi * np.sqrt(1 - rho**2)) * np.exp(-1 / (1 - rho**2) * (0.5 * x**2 + 0.5 * y **
#                                                                               2 - rho * x * y)) * (1 - np.tanh(gx + x * np.sqrt(Dx))**2) * (np.tanh(gy + y * np.sqrt(Dy)) - my)


def integrate_2DGaussian(f, args=(), Nx=50, Ny=50):
    p = np.linspace(-1, 1, Nx) * 4
    n = np.linspace(-1, 1, Ny) * 4
    P, N = np.meshgrid(p, n)
    return np.sum(f(P, N, *args)) * (p[1] - p[0]) * (n[1] - n[0])

## def fast_integrate_MFC(gx, gy, Dx, Dy, rho, Nint = 50):
##	x = np.linspace(-1, 1, Nint)*4
##	y = np.linspace(-1, 1, Nint)*4
##	X,Y = np.meshgrid(x,y)
##	z = dC(X, Y, gx, gy, Dx, Dy, rho)
##	return np.sum(z)*(x[1]-x[0])*(y[1]-y[0])
##
## def fast_integrate_MFG2(gx, gy, Dx, Dy, rho, my, Nint = 50):
##	x = np.linspace(-1, 1, Nint)*4
##	y = np.linspace(-1, 1, Nint)*4
##	X,Y = np.meshgrid(x,y)
##	z = dG2(X, Y, gx, gy, Dx, Dy, rho, my)
##	return np.sum(z)*(x[1]-x[0])*(y[1]-y[0])


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


#def dT2_10_rot(p, n, gx, gy, Dx, Dy, rho, my):
#    if n is None:
#        return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * p**2) * (1 - np.tanh(gx + p * np.sqrt(
#            1 + rho) * np.sqrt(Dx / 2))**2) * (np.tanh(gy + p * np.sqrt(1 + rho) * np.sqrt(Dy / 2)) - my)
#    elif p is None:
#        return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * p**2) * (1 - np.tanh(gx + n * np.sqrt(
#            1 - rho) * np.sqrt(Dx / 2))**2) * (np.tanh(gy - n * np.sqrt(1 - rho) * np.sqrt(Dy / 2)) - my)
#    else:
#        return 1 / (2 * np.pi) * np.exp(-0.5 * (p**2 + n**2))  \
#            * (1 - np.tanh(gx + (p * np.sqrt(1 + rho) + n * np.sqrt(1 - rho)) * np.sqrt(Dx / 2))**2) \
#            * (np.tanh(gy + (p * np.sqrt(1 + rho) - n * np.sqrt(1 - rho)) * np.sqrt(Dy / 2)) - my)


#def dT2_11_rot(p, n, gx, gy, Dx, Dy, rho):
#    if n is None:
#        return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * p**2) * (1 - np.tanh(gx + p * np.sqrt(1 + rho)
#                                                                           * np.sqrt(Dx / 2))**2) * (1 - np.tanh(gy + p * np.sqrt(1 + rho) * np.sqrt(Dy / 2))**2)
#    elif p is None:
#        return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * p**2) * (1 - np.tanh(gx + n * np.sqrt(1 - rho)
#                                                                           * np.sqrt(Dx / 2))**2) * (1 - np.tanh(gy - n * np.sqrt(1 - rho) * np.sqrt(Dy / 2))**2)
#    else:
#        return 1 / (2 * np.pi) * np.exp(-0.5 * (p**2 + n**2))  \
#            * (1 - np.tanh(gx + (p * np.sqrt(1 + rho) + n * np.sqrt(1 - rho)) * np.sqrt(Dx / 2))**2) \
#            * (1 - np.tanh(gy + (p * np.sqrt(1 + rho) - n * np.sqrt(1 - rho)) * np.sqrt(Dy / 2))**2)


#def dT2_20_rot(p, n, gx, gy, Dx, Dy, rho, my):
#    if n is None:
#        return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * p**2) * -2 * np.tanh(gx + p * np.sqrt(1 + rho) * np.sqrt(Dx / 2)) * (
#            1 - np.tanh(gx + p * np.sqrt(1 + rho) * np.sqrt(Dx / 2))**2) * (np.tanh(gy + p * np.sqrt(1 + rho) * np.sqrt(Dy / 2)) - my)
#    elif p is None:
#        return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * p**2) * -2 * np.tanh(gx + n * np.sqrt(1 - rho) * np.sqrt(Dx / 2)) * (
#            1 - np.tanh(gx + n * np.sqrt(1 - rho) * np.sqrt(Dx / 2))**2) * (np.tanh(gy - n * np.sqrt(1 - rho) * np.sqrt(Dy / 2)) - my)
#    else:
#        return 1 / (2 * np.pi) * np.exp(-0.5 * (p**2 + n**2)) * -2 * np.tanh(gx + (p * np.sqrt(1 + rho) + n * np.sqrt(1 - rho)) * np.sqrt(Dx / 2)) * (1 - np.tanh(gx + \
#                    (p * np.sqrt(1 + rho) + n * np.sqrt(1 - rho)) * np.sqrt(Dx / 2))**2) * (np.tanh(gy + (p * np.sqrt(1 + rho) - n * np.sqrt(1 - rho)) * np.sqrt(Dy / 2)) - my)


#def normal_coordinate_change(gx, gy, Dx, Dy, rho):
#    gp = gx + gy
#    gn = gx - gy
#    Dp = Dx + Dy + 2 * rho * np.sqrt(Dx * Dy)
#    Dn = Dx + Dy - 2 * rho * np.sqrt(Dx * Dy)
#    if Dn / Dp > 1E-5:
#        rhopn = (Dx - Dy) / np.sqrt(Dp * Dn)
#    else:
#        rhopn = 0
#    return gp, gn, Dp, Dn, rhopn


def update_C_P2_o1(H, J, m, m_p):
    size = len(H)
    C = np.zeros((size, size))
    g = H + np.dot(J, m_p)
    D = np.dot(J**2, 1 - m_p**2)
    rho = np.einsum(
        'i,k,ij,kj,j->ik',
        1 / np.sqrt(D),
        1 / np.sqrt(D),
        J,
        J,
        1 - m_p**2,
        optimize=True)
    if np.any(np.abs(rho) > 1):
        print(rho)
    rho = np.clip(rho, -1, 1)
    for i in range(size):
        C[i, i] = 1 - m[i]**2
        for j in range(i + 1, size):
            #			gp, gn, Dp, Dn, rhopn = normal_coordinate_change(g[i], g[j], D[i], D[j], rho[i,j])
            #			print(gp, gn, Dp, Dn, rhopn)
            #			C[i,j] = fast_integrate_MFC(g[i], g[j], D[i], D[j], rho[i,j]) - m[i]*m[j]
            if rho[i, j] > (1 - 1E5):
                C[i, j] = integrate_1DGaussian(
                    dT2_rot, (None, g[i], g[j], D[i], D[j], rho[i, j])) - m[i] * m[j]
            else:
                C[i, j] = integrate_2DGaussian(
                    dT2_rot, (g[i], g[j], D[i], D[j], rho[i, j])) - m[i] * m[j]
            C[j, i] = C[i, j]
    return C


#def update_C_P2_o2(H, J, m, m_p, V_pp):
#    size = len(H)
#    C = np.zeros((size, size))
#    G = np.zeros((size, size))
#    G10 = np.zeros((size, size))
#    G11 = np.zeros((size, size))
#    G20 = np.zeros((size, size))
#    g = H + np.dot(J, m_p)
#    D = np.dot(J**2, 1 - m_p**2)
#    print('g', g)
#    print('D', D)
#    rho = np.einsum(
#        'i,k,ij,kj,j->ik',
#        1 / np.sqrt(D),
#        1 / np.sqrt(D),
#        J,
#        J,
#        1 - m_p**2,
#        optimize=True)
#    rho[D==0] = 0
#    for i in range(size):
#        for j in range(size):
#            #			gp, gn, Dp, Dn, rhopn = normal_coordinate_change(g[i], g[j], D[i], D[j], rho[i,j])
#            if i > j:
#                if rho[i, j] > (1 - 1E5):
#                    G[i, j] = integrate_1DGaussian(
#                        dT2_rot, (None, g[i], g[j], D[i], D[j], rho[i, j])) - m[i] * m[j]
#                    G11[i, j] = integrate_1DGaussian(
#                        dT2_11_rot, (None, g[i], g[j], D[i], D[j], rho[i, j]))
#                else:
#                    G[i, j] = integrate_2DGaussian(
#                        dT2_rot, (g[i], g[j], D[i], D[j], rho[i, j])) - m[i] * m[j]
#                    G11[i, j] = integrate_2DGaussian(
#                        dT2_11_rot, (g[i], g[j], D[i], D[j], rho[i, j]))
#                G[j, i] = G[i, j]
#                G11[j, i] = G11[i, j]
#            if i != j:
#                if rho[i, j] > (1 - 1E5):
#                    G10[i, j] = integrate_1DGaussian(
#                        dT2_10_rot, (None, g[i], g[j], D[i], D[j], rho[i, j], m[j]))
#                    G20[i, j] = integrate_1DGaussian(
#                        dT2_20_rot, (None, g[i], g[j], D[i], D[j], rho[i, j], m[j]))
#                else:
#                    G10[i, j] = integrate_2DGaussian(
#                        dT2_10_rot, (g[i], g[j], D[i], D[j], rho[i, j], m[j]))
#                    G20[i, j] = integrate_2DGaussian(
#                        dT2_20_rot, (g[i], g[j], D[i], D[j], rho[i, j], m[j]))
##				integrate_rotated_2DGaussian(dG2_rot, gp, gn, Dp, Dn, rhopn,args=(m[j],))

#    for i in range(size):
#        C[i, i] = 1 - m[i]**2
#        for j in range(i + 1, size):
#            C[i, j] = G[i, j]
##			C[i,j] -= 0.5*np.sum((J[i,:]*(1 - m_p**2)*G10[i,j] + J[j,:]*(1 - m_p**2)*G10[j,i])*np.diag(V_pp))
#            C[i, j] += 0.5 * np.einsum('k,l,kl->', (1 - m_p**2)
#                                       * J[i, :], (1 - m_p**2) * J[j, :], V_pp) * G11[i, j]
#            C[i, j] += 0.5 * np.einsum('k,l,kl->', (1 - m_p**2)
#                                       * J[j, :], (1 - m_p**2) * J[i, :], V_pp) * G11[i, j]
#            C[i,
#              j] += 0.5 * 0.5 * np.einsum('k,l,kl->',
#                                          (1 - m_p**2) * J[i,
#                                                           :],
#                                          (1 - m_p**2) * J[i,
#                                                           :],
#                                          V_pp) * G20[i,
#                                                      j]
#            C[i,
#              j] += 0.5 * 0.5 * np.einsum('k,l,kl->',
#                                          (1 - m_p**2) * J[j,
#                                                           :],
#                                          (1 - m_p**2) * J[j,
#                                                           :],
#                                          V_pp) * G20[j,
#                                                      i]
#            C[j, i] = C[i, j]
##			print('C',G[i,j],np.sum((J[i,:]*(1 - m_p**2)*G[i,j] + J[j,:]*(1 - m_p**2)*G[j,i])*np.diag(V_pp)) )
##			print('G',i,j,G[i,j]+G[j,i],np.sum(np.diag(V_pp)))
##			print(C[i,j], G[i,j],C[i,j] - G[i,j] )
##			print(V_pp)
#    return C


#def update_G_P2_o2(H, J, m, m_p, Vii_pp):
#    size = len(H)
#    G = np.zeros((size, size))
#    g = H + np.dot(J, m_p)
#    D = np.dot(J**2, 1 - m_p**2)
#    rho = np.einsum(
#        'i,k,ij,kj,j->ik',
#        1 / np.sqrt(D),
#        1 / np.sqrt(D),
#        J,
#        J,
#        1 - m_p**2,
#        optimize=True)
#    for i in range(size):
#        for j in range(size):
#            #			gp, gn, Dp, Dn, rhopn = normal_coordinate_change(g[i], g[j], D[i], D[j], rho[i,j])
#            G[i, j] = integrate_2DGaussian(
#                dG2_rot, (g[i], g[j], D[i], D[j], rho[i, j], 0))
#    return G
