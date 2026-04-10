import LDCSD
from LDCSD import options
import matplotlib.pyplot as plt
import numpy as np
# Physics and parameters
X = 50
G = 3
x_mesh = np.linspace(0, 10, X+1)
e_mesh = np.linspace(0, 1000, G+1)

dE = np.diff(e_mesh)
print(dE)

s16angles, s16weights = np.polynomial.legendre.leggauss(16)

print("quadrature")
print(s16angles)
print(s16weights)


# Inputs, initialize LDCSD
m1 = LDCSD.Material(
    G = G,
    stopping_power=np.array([0.4, 0.6, 0.8]),
    stopping_power_d = np.array([0.45, 0.65, 0.85]),
    total = np.array([0.1, 3, 0.7]),
    scatter = np.array([[0.0/dE[0], 0.085/dE[1], 0.0/dE[2]],   # value is *average* over E0,
                           [0.0/dE[0], 1.0/dE[1], 2.0/dE[2]],   # integral over E1.
                           [0.0/dE[0], 0.0/dE[1], 0.3/dE[2]]]),
    # scalar_source = np.array([0.5, 0.25, 0.1])
    scalar_source = np.array([2, 0.5, 0])
)

m2 = LDCSD.Material(
    G = G,
    stopping_power=np.array([0.4, 0.6, 0.8]),
    stopping_power_d = np.array([0.45, 0.65, 0.85]),
    total = np.array([0.2, 0.8, 1.0]),
    scatter = np.array([[0.1/dE[0], 0.1/dE[1], 0.0/dE[2]],   # value is *average* over E0,
                           [0.0/dE[0], 0.45/dE[1], 0.05/dE[2]],   # integral over E1.
                           [0.0/dE[0], 0.0/dE[1], 0.5/dE[2]]]),
    # scalar_source = np.array([0.5, 0.25, 0.1])
    scalar_source = np.array([2, 0.5, 0])
)

regions = LDCSD.Regions(
    # bounds = [0, 7.0, 10],
    # materials = [m1, m2]
    bounds = [0, 3,10],
    materials = [m2,m1]
)

bcs = LDCSD.Boundaries(
    left = 0,
    right = 0
)

mesh = LDCSD.Mesh(
    x_edges = x_mesh, 
    group_bounds = e_mesh,
    mat_regions = regions,
    bounds = bcs,
    angles = s16angles,
    weights = s16weights
)

# print(mesh.x_bounds)
# print(mesh.cell_centers)

# for g in range(0, mesh.G):
#     print(f"constants, group {g}")
#     print("total:")
#     print(mesh.xs_total[g])
#     print("stopping_power:")
#     print(mesh.stopping_power[g])

# raise Exception(f"debug")

# Run LDCSD
# options = LDCSD.options(method = "high_order_transport")
LDCSD.options.scheme["method"] = "high_order_transport"
LDCSD.options.output_residuals()
x = LDCSD.high_order(mesh)



plt.figure()
I = x_mesh.size-1
for g in range(0, mesh.G):
    plot_mesh = np.array([])
    plot_up = np.array([])
    plot_down = np.array([])
    for i in range(0, I):
        plot_mesh = np.append(plot_mesh, [x_mesh[i], x_mesh[i+1]])
        plot_up = np.append(plot_up, [
            x[g, np.ravel_multi_index((i, 0, 0),(I, 2, 2))],
            x[g, np.ravel_multi_index((i, 1, 0),(I, 2, 2))]
        ])
        plot_down = np.append(plot_down, [
            x[g, np.ravel_multi_index((i, 0, 1),(I, 2, 2))],
            x[g, np.ravel_multi_index((i, 1, 1),(I, 2, 2))]
        ])

    plt.plot(plot_mesh, plot_up, label=f"g{g}, up")
    plt.plot(plot_mesh, plot_down, label=f"g{g}, down")
plt.legend()
# plt.savefig("fig.png")
plt.ylim(bottom=0)
plt.show()
