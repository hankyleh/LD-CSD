
import numpy as np

from LDCSD.opt import Options

global options
options = Options()

from LDCSD.material import Material
from LDCSD.regions import Regions
from LDCSD.boundaries import Boundaries
from LDCSD.mesh import Mesh
from LDCSD.main import high_order

# Angular Quadrature
mu = None
w = None
M = None

# Incoming flux boundary conditions
left_BC = None
right_BC = None

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

def boundary_condition(left = 0, right = 0, left_mode="vacuum", right_mode="vacuum",
                        left_isotropic=True, right_isotropic=True, 
                        left_grouped = True, right_grouped=True):

    # Assign boundary conditions: vacuum, incoming flux, reflective, or albedo
    global M, mu, w, group_bounds, G, dE, left_BC, right_BC
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
    for m in range(0, int(M/2)):
        left_BC = np.append(np.zeros(2*G), left_BC)
        right_BC = np.append(right_BC, np.zeros(2*G))
    
    

