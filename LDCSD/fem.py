import numpy
import scipy
from .mesh import Mesh

class LD_space:
    def __init__(self, 
                 mesh : Mesh, 
                 g : int,
                 lumped=False):
        self.group_bounds = mesh.group_bounds
        self.x_bounds = mesh.x_bounds
        self.Nx = self.x_bounds.size - 1
        self.g = g
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
    def index_2d(self, i, l):
        return numpy.ravel_multi_index((i, l), (self.Nx, 2))
    def flatten(self, A):
        return numpy.ravel(A)
        

class bilinear:
    def __init__(self, fespace):
        self.size = 4*fespace.Nx
        self.matrix = scipy.sparse.coo_array((self.size, self.size))
    def append(self, value, row, column):
        self.matrix.data = numpy.append(self.matrix.data, value)
        self.matrix.row = numpy.append(self.matrix.row, row)
        self.matrix.col = numpy.append(self.matrix.col, column)


class linear:
    def __init__(self, fespace):
        self.size = 4*fespace.Nx
        self.vector = numpy.zeros(self.size)
    def append(self, value, index):
        self.vector[index] += value