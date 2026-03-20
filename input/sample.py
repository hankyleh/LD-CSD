import LDCSD
import matplotlib.pyplot as plt
import numpy

# Physics and parameters
X = 10
G = 3
x_mesh = numpy.linspace(0, 10, X+1)
e_mesh = numpy.linspace(0, 1000, G+1)

# Inputs, initialize LDCSD
m1 = LDCSD.Material(
    G = G,
    stopping_power = numpy.array([2.1, 0.5]),
    total = numpy.array([1.0, 0.8]),
    scatter = numpy.array([0.8, 0.6]),
    scalar_source = numpy.array([1.0, 0.5])
)

m2 = LDCSD.Material(
    G = G,
    stopping_power=numpy.array([0.02, 0.01]),
    total=numpy.array([0.5, 0.3]),
    scatter=numpy.array([0.1, 0.05]),
    scalar_source = numpy.array([0.5, 0.25])
)

regions = LDCSD.Regions(
    bounds = [0, 7.0, 10],
    materials = [m1, m2]
)

bcs = LDCSD.Boundaries(
    left = 0,
    right = 0
)


mesh = LDCSD.Mesh(
    x_bounds = x_mesh, 
    group_bounds = e_mesh,
    regions = regions,
    boundaries = bcs
)

# Run LDCSD
LDCSD.run(mesh)

