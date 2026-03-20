import numpy
import scipy
from .mesh import Mesh

class LD_space:
    def __init__(self, 
                 mesh : Mesh, 
                 lumped=False):
        self.group_bounds = mesh.group_bounds
        self.x_bounds = mesh.x_bounds
        self.G = self.group_bounds.size - 1
        self.Nx = self.x_bounds.size - 1

        self.dx = numpy.diff(self.x_bounds)
        self.dE = numpy.diff(self.group_bounds)

        self.L  = 0.5 * numpy.array([[ 1, 1], [-1, -1]])
        self.Lb =       numpy.array([[-1, 0], [ 0,  1]])
        match lumped:
            case False:
                self.M  = (1/6) * numpy.array([[2, 1], [1, 2]])
            case True: 
                self.M  = (1/6) * numpy.array([[3, 0], [0, 3]])
            case _:
                raise Exception(f"Invalid flag lumped='{mode}' provided in fem.fespace.__init__")
    def index_2d(self, i, g):
        return numpy.ravel_multi_index((i, g), (self.Nx, self.G))
    def flatten(self, A):
        return numpy.ravel(A)
        

class bilinear:
    def __init__(self, fespace):
        self.size = 4*fespace.Nx*fespace.G
        self.A = scipy.sparse.coo_array((self.size, self.size))

class linear:
    def __init__(self, fespace):
        self.size = 4*fespace.Nx*fespace.G
        self.b = numpy.array((self.size))