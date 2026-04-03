import LDCSD
from LDCSD import options
import matplotlib.pyplot as plt
import numpy

# Physics and parameters
X = 15
G = 3
x_mesh = numpy.linspace(0, 10, X+1)
e_mesh = numpy.linspace(0, 1000, G+1)

s4angles = numpy.array([-0.3399810435848563, 0.3399810435848563, -0.8611363115940526, 0.8611363115940526])
s4weights = numpy.array([0.6521451548625461, 0.6521451548625461, 0.3478548451374538, 0.3478548451374538])

# Inputs, initialize LDCSD
m1 = LDCSD.Material(
    G = G,
    stopping_power = numpy.array([0.1, 0.2, 0.3]),
    total = numpy.array([0.3, 0.8, 0.1]),
    scatter = 0.01*numpy.array([[0.02, 0.01, 0.01],
                          [0.0, 0.01, 0.01],
                          [0.0, 0.0, 0.01]]),
    scalar_source = numpy.array([1.0, 0.5, 0.1])
)

m2 = LDCSD.Material(
    G = G,
    stopping_power=numpy.array([0.01, 0.02, 0.03]),
    total=numpy.array([0.5, 0.3, 0.1]),
    scatter = numpy.array([[0.2, 0.1, 0.1],
                          [0.0, 0.1, 0.1],
                          [0.0, 0.0, 0.1]]),
    scalar_source = numpy.array([0.5, 0.25, 0.1])
)

regions = LDCSD.Regions(
    # bounds = [0, 7.0, 10],
    # materials = [m1, m2]
    bounds = [0, 10],
    materials = [m1]
)

bcs = LDCSD.Boundaries(
    left = 0,
    right = 0
)

mesh = LDCSD.Mesh(
    x_bounds = x_mesh, 
    group_bounds = e_mesh,
    regions = regions,
    boundaries = bcs,
    angles = s4angles,
    weights = s4weights
)

# Run LDCSD
# options = LDCSD.options(method = "high_order_transport")
LDCSD.options.scheme["method"] = "high_order_transport"
LDCSD.options.output_residuals()
x = LDCSD.run(mesh)

plot_mesh = numpy.array([])
plot_up = numpy.array([])
plot_down = numpy.array([])

I = x_mesh.size-1

for i in range(0, I):
    plot_mesh = numpy.append(plot_mesh, [x_mesh[i], x_mesh[i+1]])
    plot_up = numpy.append(plot_up, [
        x[numpy.ravel_multi_index((i, 0, 0),(I, 2, 2))],
        x[numpy.ravel_multi_index((i, 1, 0),(I, 2, 2))]
    ])
    plot_down = numpy.append(plot_down, [
        x[numpy.ravel_multi_index((i, 0, 1),(I, 2, 2))],
        x[numpy.ravel_multi_index((i, 1, 1),(I, 2, 2))]
    ])

plt.figure()
plt.plot(plot_mesh, plot_up)
plt.plot(plot_mesh, plot_down)
plt.legend(["up", "down"])
# plt.savefig("fig.png")
plt.ylim(bottom=0)
plt.show()
