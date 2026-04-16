# Copyright (c) 2026, Kyle Hansen (khansen3@ncsu.edu)
#
# Supported by CARRE (https://carre-psaapiv.org/) and NCSU
#
# Licensed under BSD 3-Clause License; Redistribution and use in source and binary 
# forms, with or without modification are permitted provided that the terms of the
# license are met.

import numpy as np

from LDCSD.opt import Options

global options
options = Options()

from LDCSD.material import Material
from LDCSD.regions import Regions
from LDCSD.boundaries import Boundaries
from LDCSD.mesh import Mesh
from LDCSD.main import solve_high_order, solve_smm

# Angular Quadrature
mu = None
w = None
M = None

# Incoming flux boundary conditions
left_BC = None
right_BC = None

left_J    = None
left_phi  = None
right_J   = None
right_phi = None

# spatial mesh
x_bounds = None
I = None
regions = None
boundaries = None
cell_centers = None
dx = None

# energy group structure
group_bounds = None
G = None
dE = None

def set_quadrature(angles, weights):
    global mu, w, M
    mu = angles
    w = weights

    if weights.size != angles.size:
        raise Exception(f"Attempted to initialize mesh with {angles.size} angles and {weights.size} weights")
    M = angles.size

def set_group_struct(group_boundaries):
    global group_bounds, G, dE
    group_bounds = group_boundaries
    G = group_boundaries.size-1
    dE = (group_boundaries[1:] - group_boundaries[0:-1])

def boundary_condition(left = 0, right = 0, left_mode="vacuum", right_mode="vacuum",
                        left_isotropic=True, right_isotropic=True, 
                        left_grouped = True, right_grouped=True):

    # Assign boundary conditions: vacuum, incoming flux, reflective, or albedo
    global M, mu, w, group_bounds, G, dE, left_BC, right_BC, left_J,\
            left_phi, right_J, right_phi
    left_BC = np.zeros((int(M/2), 2*G))
    right_BC = np.zeros((int(M/2), 2*G))

    BCs = [left_BC, right_BC]
    side = [left, right]
    mode = [left_mode, right_mode]
    isotropic = [left_isotropic, right_isotropic]
    grouped = [left_grouped, right_grouped]

    # Check that modes and values do not interfere

    for i in [0, 1]:
        if mode[i] == "vacuum" or mode[i] == "vac":
            if isinstance(side[i], int):
                if (side[i] != 0):
                    raise TypeError("\n\nDo not specify BC value when mode 'vacuum' is selected\n\n")
                BCs[i] = np.zeros((int(M/2), 2*G))
            else:
                raise TypeError("\n\nDo not specify BC value when mode 'vacuum' is selected\n\n")
        elif mode[i] == "incoming":
            if not isinstance(side[i], np.ndarray):
                raise TypeError("\n\nMust specify incoming flux BCs as numpy array\n\n")

            if isotropic[i] == True:
                if grouped[i] == True:
                    if side[i].shape != (G,):
                        raise ValueError("\n\nProvide G isotropic incoming flux values for incoming flux condition\n\n")
                    for m in range(0, int(M/2)):
                        BCs[i][m] = np.repeat(side[i], 2)
                else:
                    # BC given in energy corner values
                    if side[i].shape != (2*G,):
                        raise ValueError("\n\nProvide 2G isotropic incoming flux values\n\n")
                    for m in range(0, int(M/2)):
                        BCs[i][m, :] = side[i]
            else:
                # BC is anisotropic
                if grouped[i] == True: 
                    if side[i].shape != (int(M/2), G):
                        raise ValueError("\n\nProvide (M/2, G) isotropic incoming flux values\n\n")
                    for m in range(0, int(M/2)):
                        BCs[i][m, :] = np.repeat(side[i, m], 2)
                else:
                    # BC given in energy corner values
                    if side[i].shape != (int(M/2), 2*G):
                        raise ValueError("\n\nProvide (M/2, 2G) isotropic incoming flux values\n\n")
                    for m in range(0, int(M/2)):
                        BCs[i][m] = side[i, m]
        elif mode[i] == "reflective" or mode[i] == "refl":
            raise ValueError("\n\nReflective BC not supported yet\n\n")
        elif mode[i] == "albedo":
            raise ValueError("\n\nAlbedo BC not supported yet\n\n")
        else:
            raise ValueError("Unrecognized boundary condition")
    
    tmp_left = np.zeros((M, 2*G))
    tmp_right = np.zeros((M, 2*G))

    tmp_left[(int(M/2)):, :] = left_BC
    tmp_right[:int(M/2), :] = right_BC

    J_left   = np.zeros((2*G))
    phi_left = np.zeros((2*G))

    J_right   = np.zeros((2*G))
    phi_right = np.zeros((2*G))

    for m in range(0, M):
        J_left    += tmp_left[m] * mu[m] * w[m]
        phi_left  += tmp_left[m]  * w[m]
        J_right   += tmp_right[m] * mu[m] * w[m]
        phi_right += tmp_right[m]  * w[m]

    left_BC = tmp_left
    right_BC = tmp_right

    left_J    = J_left    
    left_phi  = phi_left  
    right_J   = J_right   
    right_phi = phi_right 
    
    

