import LDCSD
from LDCSD import options
import matplotlib.pyplot as plt
import numpy as np


# spatial discretization
nx = 35
X = 10
x_mesh = np.linspace(0, X, nx+1)

# angular discretization
gauss_leg_angles, gauss_leg_weights = np.polynomial.legendre.leggauss(8)
LDCSD.set_quadrature(gauss_leg_angles, gauss_leg_weights)

# energy discretization
G = 3
e_mesh = np.linspace(0, 1000, G+1)
dE = np.diff(e_mesh)

LDCSD.set_group_struct(e_mesh)

# problem parameters
m1 = LDCSD.Material(
    G = G,
    stopping_power=np.array([0.4, 0.6, 0.8])*0,
    stopping_power_d = np.array([0.45, 0.65, 0.85])*0,
    total = np.array([1, 5, 10]),
    scatter = 0*np.array([[0.2/dE[0], 0.8/dE[1], 0.0/dE[2]],   # value is *average* over E0,
                           [0.0/dE[0], 1.0/dE[1], 2.0/dE[2]],   # integral over E1.
                           [0.0/dE[0], 0.0/dE[1], 7.0/dE[2]]]),
    scalar_source = np.array([0, 0, 0])
)

regions = LDCSD.Regions(
    # bounds = [0, 7.0, 10],
    # materials = [m1, m2]
    bounds = [0, X],
    materials = [m1]
)

LDCSD.boundary_condition(left=np.array([1.0, 0, 0]), 
                         right=np.array([0.3, 0, 0]),
                         left_mode = "incoming",
                         right_mode = "incoming")




mesh = LDCSD.Mesh(
    x_edges = x_mesh, 
    group_boundaries = e_mesh,
    mat_regions = regions,
    angles = gauss_leg_angles,
    weights = gauss_leg_weights
)











LDCSD.options.scheme["method"] = "high_order_transport"
LDCSD.options.output_residuals()




scalar = np.zeros((LDCSD.G, 4*LDCSD.I))
angular = np.zeros((LDCSD.G, LDCSD.M, 4*LDCSD.I))

LDCSD.solve_smm(mesh, scalar, angular)
