from .fem import *
from .mesh import Mesh
import numpy
import scipy

def solve_DO(mesh : Mesh, mu : float):
    # space-energy mesh sweep for one Discrete Ordinate (DO)

    print("Executing 'solve_DO'")

    # Initialize Bilinear and Linear forms
    fespace = LD_space(mesh)
    A = bilinear(fespace)
    b = linear(fespace)

    print(f"Bilinear Form: {A.size}x{A.size} matrix")
    print(f"Linear Form: {b.size}x1 vector")

    # Boundary group, boundary cells (g=0, i=0,Nx-1)
    # Energy and space upwind= psi_in_(g/i)
    g=0
    i=0

    

    # Boundary group, interior spatial cells (g=0, i=1,2..Nx-2)
    # energy upwind= psi_in_g
    
    # Boundary cells i=0 and Nx-1, interior groups
    # space upwind= psi_in_i

    # Interior cells


    # Solve Linear system


    return 0

# def sweep_DO():
    # sweep in energy. Solve 1d FEM system at each energy group