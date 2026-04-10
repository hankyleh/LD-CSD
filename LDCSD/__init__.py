
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


x_bounds = None
I = None
regions = None
boundaries = None
cell_centers = None
dx = None
