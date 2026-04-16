# Copyright (c) 2026, Kyle Hansen (khansen3@ncsu.edu)
#
# Supported by CARRE (https://carre-psaapiv.org/) and NCSU
#
# Licensed under BSD 3-Clause License; Redistribution and use in source and binary 
# forms, with or without modification are permitted provided that the terms of the
# license are met.

import LDCSD
from LDCSD.fem import bilinear

import numpy as np
import scipy.sparse as sparse



def compute_q0(source):
    # compute zeroth angular moment of some external source q
    # source must have shape (M, dx)
    q0 = np.zeros((source.shape[1]))
    for m in range(0, LDCSD.M):
        q0 += source[m] * LDCSD.w[m]
    return q0


def compute_q1(source):
    # compute first angular moment of some external source q
    # source must have shape (M, dx)
    q1 = np.zeros((source.shape[1]))
    for m in range(0, LDCSD.M):
        q1 += source[m] * (LDCSD.w[m] * LDCSD.mu[m])
    return q1


def compute_scatter_source(source, g, soln_g, soln, xs_scatter, mesh):

    # compute scattering source RHS for SM equations

    # result: write scattering source to 'scatter_source'

    # [ phi ]
    # [ phi ]
    # [  J  ]
    # [  J  ]

    I = LDCSD.I
    J = 4*LDCSD.I

    dx = LDCSD.dx

    for gp in range(0, LDCSD.G):
        solvec = soln[gp]
        if gp == g:
            solvec = soln_g

        for i in range(0, LDCSD.I):
            ind_L_d = np.ravel_multi_index((i, 0, 0), (I, 2, 2))
            ind_L_u = np.ravel_multi_index((i, 0, 1), (I, 2, 2))
            ind_R_d = np.ravel_multi_index((i, 1, 0), (I, 2, 2))
            ind_R_u = np.ravel_multi_index((i, 1, 1), (I, 2, 2))

            # BALANCE EQN
            # L, u
            source[ind_L_u] += (dx[i]*LDCSD.dE[gp]/24)*xs_scatter[gp, g, i]*(
                2*solvec[ind_L_d] + solvec[ind_R_d] 
                +2*solvec[ind_L_u] + solvec[ind_R_u]
            )
            # R, u
            source[ind_R_u] += (dx[i]*LDCSD.dE[gp]/24)*xs_scatter[gp, g, i]*(
                solvec[ind_L_d] + 2*solvec[ind_R_d] 
                +solvec[ind_L_u] + 2*solvec[ind_R_u]
            )
            # L, d
            source[ind_L_d] += (dx[i]*LDCSD.dE[gp]/24)*xs_scatter[gp, g, i]*(
                2*solvec[ind_L_d] + solvec[ind_R_d] 
                +2*solvec[ind_L_u] + solvec[ind_R_u]
            )
            # R, d
            source[ind_R_d] += (dx[i]*LDCSD.dE[gp]/24)*xs_scatter[gp, g, i]*(
                solvec[ind_L_d] + 2*solvec[ind_R_d] 
                +solvec[ind_L_u] + 2*solvec[ind_R_u]
            )


            # FIRST MOMENT EQUATION
            # L, u
            source[J+ind_L_u] += (dx[i]*LDCSD.dE[gp]/24)*xs_scatter[gp, g, i]*(
                2*solvec[J+ind_L_d] + solvec[J+ind_R_d] 
                +2*solvec[J+ind_L_u] + solvec[J+ind_R_u]
            )
            # R, u
            source[J+ind_R_u] += (dx[i]*LDCSD.dE[gp]/24)*xs_scatter[gp, g, i]*(
                solvec[J+ind_L_d] + 2*solvec[J+ind_R_d] 
                +solvec[J+ind_L_u] + 2*solvec[J+ind_R_u]
            )
            # L, d
            source[J+ind_L_d] += (dx[i]*LDCSD.dE[gp]/24)*xs_scatter[gp, g, i]*(
                2*solvec[J+ind_L_d] + solvec[J+ind_R_d] 
                +2*solvec[J+ind_L_u] + solvec[J+ind_R_u]
            )
            # R, d
            source[J+ind_R_d] += (dx[i]*LDCSD.dE[gp]/24)*xs_scatter[gp, g, i]*(
                solvec[J+ind_L_d] + 2*solvec[J+ind_R_d] 
                +solvec[J+ind_L_u] + 2*solvec[J+ind_R_u]
            )

def form_LHS(A : bilinear, 
             dx: np.ndarray, 
             sigma : np.ndarray, 
             S: np.ndarray, 
             S_d : np.ndarray,
             dEg : float):
    
    # forms LD stiffness matrix of form

    # balance equation[        ][ phi ]
    # balance equation[        ][ phi ]
    # first moment eqn[        ][  J  ]
    # first moment eqn[        ][  J  ]

    I = LDCSD.I
    J = 4*I

    # interior cells
    for i in range(0, I):
        if i > 0:
            ind_R_d_b = np.ravel_multi_index((i-1, 1, 0), (I, 2, 2))
            ind_R_u_b = np.ravel_multi_index((i-1, 1, 1), (I, 2, 2))
        ind_L_d = np.ravel_multi_index((i, 0, 0), (I, 2, 2))
        ind_L_u = np.ravel_multi_index((i, 0, 1), (I, 2, 2))
        ind_R_d = np.ravel_multi_index((i, 1, 0), (I, 2, 2))
        ind_R_u = np.ravel_multi_index((i, 1, 1), (I, 2, 2))
        if i < I-1:
            ind_L_d_b = np.ravel_multi_index((i+1, 0, 0), (I, 2, 2))
            ind_L_u_b = np.ravel_multi_index((i+1, 0, 1), (I, 2, 2))

        # BALANCE LU
        # streaming
        # interface
        if i>0:
            A.append(-(2*(1/6)*(1/2)), ind_L_u, J+ind_R_u_b)
            A.append(-(  (1/6)*(1/2)), ind_L_u, J+ind_R_d_b)
            A.append(-(2*(1/6)*(1/4)), ind_L_u, ind_R_u_b)
            A.append(-(  (1/6)*(1/4)), ind_L_u, ind_R_d_b)
        A.append(-(2*(1/6)*(1/2)), ind_L_u, J+ind_L_u  )
        A.append(-(  (1/6)*(1/2)), ind_L_u, J+ind_L_d  )
        A.append( (2*(1/6)*(1/4)), ind_L_u, ind_L_u  )
        A.append( (  (1/6)*(1/4)), ind_L_u, ind_L_d  )
        
        # within-cell
        A.append(  (1/12), ind_L_u, J+ind_L_d)
        A.append(  (1/12), ind_L_u, J+ind_R_d)
        A.append(2*(1/12), ind_L_u, J+ind_L_u)
        A.append(2*(1/12), ind_L_u, J+ind_R_u)
        # absorption
        A.append(  (2*(1/6)*(1/6)*sigma[i]*dx[i]), ind_L_u, ind_L_d)
        A.append(  (  (1/6)*(1/6)*sigma[i]*dx[i]), ind_L_u, ind_R_d)
        A.append(2*(2*(1/6)*(1/6)*sigma[i]*dx[i]), ind_L_u, ind_L_u)
        A.append(2*(  (1/6)*(1/6)*sigma[i]*dx[i]), ind_L_u, ind_R_u)
        # CSD

        # BALANCE RU
        # streaming
        # interface
        if i < I-1:
            A.append((2*(1/6)*(1/2)), ind_R_u, J+ind_L_u_b)
            A.append((  (1/6)*(1/2)), ind_R_u, J+ind_L_d_b)
            A.append(-(2*(1/6)*(1/4)), ind_R_u, ind_L_u_b)
            A.append(-(  (1/6)*(1/4)), ind_R_u, ind_L_d_b)
        A.append((2*(1/6)*(1/2)), ind_R_u, J+ind_R_u)
        A.append((  (1/6)*(1/2)), ind_R_u, J+ind_R_d)
        A.append( (2*(1/6)*(1/4)), ind_R_u, ind_R_u)
        A.append( (  (1/6)*(1/4)), ind_R_u, ind_R_d)
        # within-cell
        A.append(  (-1/12), ind_R_u, J+ind_L_d)
        A.append(  (-1/12), ind_R_u, J+ind_R_d)
        A.append(2*(-1/12), ind_R_u, J+ind_L_u)
        A.append(2*(-1/12), ind_R_u, J+ind_R_u)
        # absorption
        A.append(  (  (1/6)*(1/6)*sigma[i]*dx[i]), ind_R_u, ind_L_d)
        A.append(  (2*(1/6)*(1/6)*sigma[i]*dx[i]), ind_R_u, ind_R_d)
        A.append(2*(  (1/6)*(1/6)*sigma[i]*dx[i]), ind_R_u, ind_L_u)
        A.append(2*(2*(1/6)*(1/6)*sigma[i]*dx[i]), ind_R_u, ind_R_u)
        # CSD
        
        # BALANCE LD
        # streaming
        # interface
        if i>0:
            A.append(-(  (1/6)*(1/2)), ind_L_d, J+ind_R_u_b)
            A.append(-(2*(1/6)*(1/2)), ind_L_d, J+ind_R_d_b)
            A.append(-(  (1/6)*(1/4)), ind_L_d, ind_R_u_b)
            A.append(-(2*(1/6)*(1/4)), ind_L_d, ind_R_d_b)
        A.append(-(  (1/6)*(1/2)), ind_L_d, J+ind_L_u  )
        A.append(-(2*(1/6)*(1/2)), ind_L_d, J+ind_L_d  )
        A.append( (  (1/6)*(1/4)), ind_L_d, ind_L_u  )
        A.append( (2*(1/6)*(1/4)), ind_L_d, ind_L_d  )
        # within-cell
        A.append(2*(1/12), ind_L_d, J+ind_L_d)
        A.append(2*(1/12), ind_L_d, J+ind_R_d)
        A.append(  (1/12), ind_L_d, J+ind_L_u)
        A.append(  (1/12), ind_L_d, J+ind_R_u)
        # absorption
        A.append(2*(2*(1/6)*(1/6)*sigma[i]*dx[i]), ind_L_d, ind_L_d)
        A.append(2*(  (1/6)*(1/6)*sigma[i]*dx[i]), ind_L_d, ind_R_d)
        A.append(  (2*(1/6)*(1/6)*sigma[i]*dx[i]), ind_L_d, ind_L_u)
        A.append(  (  (1/6)*(1/6)*sigma[i]*dx[i]), ind_L_d, ind_R_u)
        # CSD
        
        # BALANCE RD
        # streaming
        # interface
        if i < I-1:
            A.append((  (1/6)*(1/2)), ind_R_d, J+ind_L_u_b)
            A.append((2*(1/6)*(1/2)), ind_R_d, J+ind_L_d_b)
            A.append(-(  (1/6)*(1/4)), ind_R_d, ind_L_u_b)
            A.append(-(2*(1/6)*(1/4)), ind_R_d, ind_L_d_b)
        A.append((  (1/6)*(1/2)), ind_R_d, J+ind_R_u)
        A.append((2*(1/6)*(1/2)), ind_R_d, J+ind_R_d)
        A.append( (  (1/6)*(1/4)), ind_R_d, ind_R_u)
        A.append( (2*(1/6)*(1/4)), ind_R_d, ind_R_d)
        # within-cell
        A.append(2*(-1/12), ind_R_d, J+ind_L_d)
        A.append(2*(-1/12), ind_R_d, J+ind_R_d)
        A.append(  (-1/12), ind_R_d, J+ind_L_u)
        A.append(  (-1/12), ind_R_d, J+ind_R_u)
        # absorption
        A.append(2*(  (1/6)*(1/6)*sigma[i]*dx[i]), ind_R_d, ind_L_d)
        A.append(2*(2*(1/6)*(1/6)*sigma[i]*dx[i]), ind_R_d, ind_R_d)
        A.append(  (  (1/6)*(1/6)*sigma[i]*dx[i]), ind_R_d, ind_L_u)
        A.append(  (2*(1/6)*(1/6)*sigma[i]*dx[i]), ind_R_d, ind_R_u)
        # CSD


        # FIRST MOMENT LU
        # streaming
        # interface
        if i>0:
            A.append(-(2*(1/18)*(3/4)), J+ind_L_u, J+ind_R_u_b)
            A.append(-(  (1/18)*(3/4)), J+ind_L_u, J+ind_R_d_b)
            A.append(-(2*(1/18)*(1/2)), J+ind_L_u, ind_R_u_b)
            A.append(-(  (1/18)*(1/2)), J+ind_L_u, ind_R_d_b)
        A.append( (2*(1/18)*(3/4)), J+ind_L_u, J+ind_L_u  )
        A.append( (  (1/18)*(3/4)), J+ind_L_u, J+ind_L_d  )
        A.append(-(2*(1/18)*(1/2)), J+ind_L_u, ind_L_u  )
        A.append(-(  (1/18)*(1/2)), J+ind_L_u, ind_L_d  )
        # within-cell
        A.append(  (1/36), J+ind_L_u, ind_L_d)
        A.append(  (1/36), J+ind_L_u, ind_R_d)
        A.append(2*(1/36), J+ind_L_u, ind_L_u)
        A.append(2*(1/36), J+ind_L_u, ind_R_u)
        # absorption
        A.append(  (2*(1/6)*(1/6)*sigma[i]*dx[i]), J+ind_L_u, J+ind_L_d)
        A.append(  (  (1/6)*(1/6)*sigma[i]*dx[i]), J+ind_L_u, J+ind_R_d)
        A.append(2*(2*(1/6)*(1/6)*sigma[i]*dx[i]), J+ind_L_u, J+ind_L_u)
        A.append(2*(  (1/6)*(1/6)*sigma[i]*dx[i]), J+ind_L_u, J+ind_R_u)
        # CSD
        
        # FIRST MOMENT RU
        # streaming
        # interface
        if i < I-1:
            A.append(-(2*(1/18)*(3/4)), J+ind_R_u, J+ind_L_u_b)
            A.append(-(  (1/18)*(3/4)), J+ind_R_u, J+ind_L_d_b)
            A.append((2*(1/18)*(1/2)), J+ind_R_u, ind_L_u_b)
            A.append((  (1/18)*(1/2)), J+ind_R_u, ind_L_d_b)
        A.append( (2*(1/18)*(3/4)), J+ind_R_u, J+ind_R_u)
        A.append( (  (1/18)*(3/4)), J+ind_R_u, J+ind_R_d)
        A.append((2*(1/18)*(1/2)), J+ind_R_u, ind_R_u)
        A.append((  (1/18)*(1/2)), J+ind_R_u, ind_R_d)
        # within-cell
        A.append(  (-1/36), J+ind_R_u, ind_L_d)
        A.append(  (-1/36), J+ind_R_u, ind_R_d)
        A.append(2*(-1/36), J+ind_R_u, ind_L_u)
        A.append(2*(-1/36), J+ind_R_u, ind_R_u)
        # absorption
        A.append(  (  (1/6)*(1/6)*sigma[i]*dx[i]), J+ind_R_u, J+ind_L_d)
        A.append(  (2*(1/6)*(1/6)*sigma[i]*dx[i]), J+ind_R_u, J+ind_R_d)
        A.append(2*(  (1/6)*(1/6)*sigma[i]*dx[i]), J+ind_R_u, J+ind_L_u)
        A.append(2*(2*(1/6)*(1/6)*sigma[i]*dx[i]), J+ind_R_u, J+ind_R_u)
        # CSD
        
        # FIRST MOMENT LD
        # streaming
        # interface
        if i>0:
            A.append(-(  (1/18)*(3/4)), J+ind_L_d, J+ind_R_u_b)
            A.append(-(2*(1/18)*(3/4)), J+ind_L_d, J+ind_R_d_b)
            A.append(-(  (1/18)*(1/2)), J+ind_L_d, ind_R_u_b)
            A.append(-(2*(1/18)*(1/2)), J+ind_L_d, ind_R_d_b)
        A.append( (  (1/18)*(3/4)), J+ind_L_d, J+ind_L_u  )
        A.append( (2*(1/18)*(3/4)), J+ind_L_d, J+ind_L_d  )
        A.append(-(  (1/18)*(1/2)), J+ind_L_d, ind_L_u  )
        A.append(-(2*(1/18)*(1/2)), J+ind_L_d, ind_L_d  )
        # within-cell
        A.append(2*(1/36), J+ind_L_d, ind_L_d)
        A.append(2*(1/36), J+ind_L_d, ind_R_d)
        A.append(  (1/36), J+ind_L_d, ind_L_u)
        A.append(  (1/36), J+ind_L_d, ind_R_u)
        # absorption
        A.append(2*(2*(1/6)*(1/6)*sigma[i]*dx[i]), J+ind_L_d, J+ind_L_d)
        A.append(2*(  (1/6)*(1/6)*sigma[i]*dx[i]), J+ind_L_d, J+ind_R_d)
        A.append(  (2*(1/6)*(1/6)*sigma[i]*dx[i]), J+ind_L_d, J+ind_L_u)
        A.append(  (  (1/6)*(1/6)*sigma[i]*dx[i]), J+ind_L_d, J+ind_R_u)
        # CSD
        
        # FIRST MOMENT RD
        # streaming
        # interface
        if i < I-1:
            A.append(-(  (1/18)*(3/4)), J+ind_R_d, J+ind_L_u_b)
            A.append(-(2*(1/18)*(3/4)), J+ind_R_d, J+ind_L_d_b)
            A.append((  (1/18)*(1/2)), J+ind_R_d, ind_L_u_b)
            A.append((2*(1/18)*(1/2)), J+ind_R_d, ind_L_d_b)
        A.append( (  (1/18)*(3/4)), J+ind_R_d, J+ind_R_u)
        A.append( (2*(1/18)*(3/4)), J+ind_R_d, J+ind_R_d)
        A.append((  (1/18)*(1/2)), J+ind_R_d, ind_R_u)
        A.append((2*(1/18)*(1/2)), J+ind_R_d, ind_R_d)
        # within-cell
        A.append(2*(-1/36), J+ind_R_d, ind_L_d)
        A.append(2*(-1/36), J+ind_R_d, ind_R_d)
        A.append(  (-1/36), J+ind_R_d, ind_L_u)
        A.append(  (-1/36), J+ind_R_d, ind_R_u)
        # absorption
        A.append(2*(  (1/6)*(1/6)*sigma[i]*dx[i]), J+ind_R_d, J+ind_L_d)
        A.append(2*(2*(1/6)*(1/6)*sigma[i]*dx[i]), J+ind_R_d, J+ind_R_d)
        A.append(  (  (1/6)*(1/6)*sigma[i]*dx[i]), J+ind_R_d, J+ind_L_u)
        A.append(  (2*(1/6)*(1/6)*sigma[i]*dx[i]), J+ind_R_d, J+ind_R_u)
        # CSD
    


def form_RHS(b,
             q0             : np.ndarray, 
             q1             : np.ndarray, 
             scatter_source : np.ndarray, 
             F_closure      : np.ndarray, 
             F_bound        : np.ndarray,
             k_intf         : np.ndarray,
             t_intf         : np.ndarray,
             upwind_soln    : np.ndarray,
             S_bound        : np.ndarray,
             dx             : np.ndarray, 
             dEg            : float,
             g              : int):

    I = LDCSD.I
    J = 4*I

    # interior cells
    for i in range(0, I):
        ind_L_d = np.ravel_multi_index((i, 0, 0), (I, 2, 2))
        ind_L_u = np.ravel_multi_index((i, 0, 1), (I, 2, 2))
        ind_R_d = np.ravel_multi_index((i, 1, 0), (I, 2, 2))
        ind_R_u = np.ravel_multi_index((i, 1, 1), (I, 2, 2))

        left_u = np.ravel_multi_index((i, 1), (LDCSD.I+1, 2))
        left_d = np.ravel_multi_index((i, 0), (LDCSD.I+1, 2))
        right_u = np.ravel_multi_index((i+1, 1), (LDCSD.I+1, 2))
        right_d = np.ravel_multi_index((i+1, 0), (LDCSD.I+1, 2))
        
        # BALANCE LU
        # scattering source
        # external source
        b.append((dx[i]/36)*(4*q0[ind_L_u] + 2*q0[ind_R_u] 
                          + 2*q0[ind_L_d] + q0[ind_R_d]), ind_L_u)
        # interface
        b.append((1/6)*(2*(k_intf[left_u]) + k_intf[left_d]), ind_L_u)
        # CSD

        # BALANCE RU
        # scattering source
        # external source
        b.append((dx[i]/36)*(4*q0[ind_R_u] + 2*q0[ind_R_d] 
                          + 2*q0[ind_L_u] + q0[ind_L_d]), ind_R_u)
        # interface
        b.append(-(1/6)*(2*(k_intf[right_u]) + k_intf[right_d]), ind_R_u)
        # CSD
        
        # BALANCE LD
        # scattering source
        # external source
        b.append((dx[i]/36)*(4*q0[ind_L_d] + 2*q0[ind_L_u] 
                          + 2*q0[ind_R_d] + q0[ind_R_u]), ind_L_d)
        # interface
        b.append((1/6)*((k_intf[left_u]) + 2*k_intf[left_d]), ind_L_d)
        
        # BALANCE RD
        # scattering source
        # external source
        b.append((dx[i]/36)*(4*q0[ind_R_d] + 2*q0[ind_R_u] 
                          + 2*q0[ind_L_d] + q0[ind_L_u]), ind_R_d)
        # interface
        b.append(-(1/6)*((k_intf[right_u]) + 2*k_intf[right_d]), ind_R_d)
        

        # FIRST MOMENT LU
        # scattering source
        # external source
        b.append((1/6)*(4*q1[ind_L_u] + 2*q1[ind_L_d] 
                          + 2*q1[ind_R_u] + q1[ind_R_d]), J+ind_L_u)
        # CSD
        # interface
        b.append((1/18)*(2*t_intf[left_u] + t_intf[left_d]), J+ind_L_u)
        # closure
        b.append(-(1/6)*(2*F_bound[left_u] + F_bound[left_d]), J+ind_L_u)
        b.append((1/12)*(2*(F_closure[ind_L_u] + F_closure[ind_R_u]) +
                           (F_closure[ind_L_d] + F_closure[ind_R_d])), J+ind_L_u)

        # FIRST MOMENT RU
        # scattering source
        # external source
        b.append((1/6)*(4*q1[ind_R_u] + 2*q1[ind_R_d] 
                          + 2*q1[ind_L_u] + q1[ind_L_d]), J+ind_R_u)
        # CSD
        # interface
        b.append(-(1/18)*(2*t_intf[right_u] + t_intf[right_d]), J+ind_R_u)
        # closure
        b.append((1/6)*(2*F_bound[right_u] + F_bound[right_d]), J+ind_R_u)
        b.append((-1/12)*(2*(F_closure[ind_L_u] + F_closure[ind_R_u]) +
                           (F_closure[ind_L_d] + F_closure[ind_R_d])), J+ind_R_u)
        
        # FIRST MOMENT LD
        # scattering source
        # external source
        b.append((1/6)*(4*q1[ind_L_d] + 2*q1[ind_L_u] 
                          + 2*q1[ind_R_d] + q1[ind_R_u]), J+ind_L_d)
        # interface
        b.append((1/18)*(t_intf[left_u] + 2*t_intf[left_d]), J+ind_L_d)
        # closure
        b.append(-(1/6)*(F_bound[left_u] + 2*F_bound[left_d]), J+ind_L_d)
        b.append((1/12)*(  (F_closure[ind_L_u] + F_closure[ind_R_u]) +
                         2*(F_closure[ind_L_d] + F_closure[ind_R_d])), J+ind_L_d)
        
        # FIRST MOMENT RD
        # scattering source
        # external source
        b.append((1/6)*(4*q1[ind_R_d] + 2*q1[ind_R_u] 
                          + 2*q1[ind_L_d] + q1[ind_L_u]), J+ind_R_d)
        # interface
        b.append(-(1/18)*(t_intf[right_u] + 2*t_intf[right_d]), J+ind_R_d)
        # closure
        b.append((1/6)*(F_bound[right_u] + 2*F_bound[right_d]), J+ind_R_d)
        b.append((-1/12)*(  (F_closure[ind_L_u] + F_closure[ind_R_u]) +
                          2*(F_closure[ind_L_d] + F_closure[ind_R_d])), J+ind_R_d)
        
    # Left boundary condition
    i=0
    ind_L_d = np.ravel_multi_index((i, 0, 0), (I, 2, 2))
    ind_L_u = np.ravel_multi_index((i, 0, 1), (I, 2, 2))

    b.append((1/6)*(LDCSD.left_J[(2*g)] + 2*LDCSD.left_J[(2*g)+1]), ind_L_u)
    b.append((1/6)*(2*LDCSD.left_J[(2*g)] + LDCSD.left_J[(2*g)+1]),   ind_L_d)
    b.append((1/18)*(LDCSD.left_phi[(2*g)] + 2*LDCSD.left_phi[(2*g)+1]), J+ind_L_u)
    b.append((1/18)*(2*LDCSD.left_phi[(2*g)] + LDCSD.left_phi[(2*g)+1]), J+ind_L_d)


    # Right boundary condition
    i = LDCSD.I - 1
    ind_R_d = np.ravel_multi_index((i, 1, 0), (I, 2, 2))
    ind_R_u = np.ravel_multi_index((i, 1, 1), (I, 2, 2))

    b.append(-(1/6)*(2*LDCSD.right_J[(2*g)] + LDCSD.right_J[(2*g)+1]),   ind_R_d)
    b.append(-(1/6)*(LDCSD.right_J[(2*g)] + 2*LDCSD.right_J[(2*g)+1]),   ind_R_u)
    b.append(-(1/18)*(2*LDCSD.right_phi[(2*g)] + LDCSD.right_phi[(2*g)+1]), J+ind_R_d)
    b.append(-(1/18)*(LDCSD.right_phi[(2*g)] + 2*LDCSD.right_phi[(2*g)+1]), J+ind_R_u)


def solve_p1(scalar, current):
    # solve linear system of SMM equations for a single energy group

    # Result: write scalar to 'scalar' and current to 'current'

    # form LHS (stiffness matrix)
    # form RHS (load vector)

    # vec = solve(...)

    # scalar, current = unpack(vec)
    pass