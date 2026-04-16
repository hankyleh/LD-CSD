# Copyright (c) 2026, Kyle Hansen (khansen3@ncsu.edu)
#
# Supported by CARRE (https://carre-psaapiv.org/) and NCSU
#
# Licensed under BSD 3-Clause License; Redistribution and use in source and binary 
# forms, with or without modification are permitted provided that the terms of the
# license are met.

from LDCSD import mesh, regions, boundaries, transport, smm
from LDCSD import options

# from LDCSD.smm import form_LHS

from LDCSD.fem import *
import LDCSD
# from LDCSD.options import options
import numpy as np
import scipy.sparse as sparse

import matplotlib.pyplot as plt


def solve_high_order(mesh):
    # run high-order iteration scheme here
    scalar, angular = transport.scattering_iteration(mesh)
    return scalar, angular


def solve_smm(mesh, scalar, angular):
    # Solve using LOSM equations

    # begin with guess for initial scalar flux, and until tolerance is met
    # perform ONE energy-angle sweep of high-order transport
    # compute closures
    # Solve low-order system for scalar flux and current
    # pass scalar flux onto next iteration

    # Result:
    # write solution to 'scalar', 'angular'

    tol = 1e-5
    max_iters = 1
    print("TOLERANCE HARD-CODXED IN MAIN.SMM")

    old_scalar = scalar.copy()
    old_angular = angular.copy()



    fespace = LD_space(mesh, 0)

    F_closure = np.zeros((4*LDCSD.I))
    F_bound = np.zeros((2*(LDCSD.I+1)))
    T_plus = np.zeros((LDCSD.I, 2))
    T_minus = np.zeros((LDCSD.I, 2))
    K_plus = np.zeros((LDCSD.I, 2))
    K_minus = np.zeros((LDCSD.I, 2))
    scatter_source_LOSM = np.zeros((8*LDCSD.I))
    losm_solution = np.zeros((LDCSD.G, 8*LDCSD.I))
    n=0
    change = 1
    while (n < max_iters) and (change > tol):
        n+=1

        # one high-order sweep
        for g in range(0, LDCSD.G):
            if g==0:
                upwind_e_flux = np.zeros((LDCSD.M, 4*LDCSD.I))
            else:
                upwind_e_flux = angular[g-1]
            scatter_source = transport.compute_scatter_source(g, old_scalar[g], old_scalar,
                                                              mesh.xs_scatter, mesh)
            _, angular[g] = transport.sweep(g, mesh, mesh.xs_total[g], 
                                            mesh.stopping_power[g],
                                            mesh.stopping_power_d[g],
                                            mesh.stopping_power_d[g-1],
                                            scatter_source,
                                            upwind_e_flux,
                                            mesh.angular_source[g])
        
        upwind = np.zeros(8*LDCSD.I)
        for g in range(0, LDCSD.G):
            # compute closures
            transport.calculate_closure(g, F_closure, F_bound, angular[g])
            T_total = transport.calculate_t_bdry(T_plus, T_minus, angular[g])
            K_total = transport.calculate_k_bdry(K_plus, K_minus, angular[g])

            A = bilinear(8*LDCSD.I)
            b = linear(8*LDCSD.I)
            smm.form_LHS(A, LDCSD.dx, mesh.xs_total[g], mesh.stopping_power[g], mesh.stopping_power_d[g], LDCSD.dE[g])

            q0 = smm.compute_q0(mesh.angular_source[g])
            q1 = smm.compute_q1(mesh.angular_source[g])

            scatter_source_LOSM = np.zeros((8*LDCSD.I))
            # smm.compute_scatter_source(scatter_source_LOSM, g, losm_solution[g], 
            #                            losm_solution, mesh.xs_scatter, mesh)
        
            smm.form_RHS(b, q0, q1, scatter_source_LOSM, F_closure, F_bound, K_total, T_total, 
                         upwind, mesh.stopping_power_d[g-1], LDCSD.dx, LDCSD.dE[g], g)
            
            x = sparse.linalg.spsolve(A.matrix.tocsr(), b.vector)
            print(x)