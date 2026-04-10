from .fem import *
from .mesh import Mesh
import numpy as np
import scipy

from LDCSD import options

def form_HO_LHS(A, mu, dx, dE, I, xs_total_g, stop_power_g, stop_power_bound, M):
    
    # coefficients for absorption + streaming equations
    v1 = dx*((1/6)*(xs_total_g) + (1/(2*dE))*(stop_power_g))
    v2 = dx*((1/3)*(xs_total_g) + (1/(2*dE))*(stop_power_g))
    v3 = dx*((1/3)*(xs_total_g) + (1/dE)*((-1/2)*stop_power_g + stop_power_bound))
    v4 = dx*((1/6)*(xs_total_g) + (1/dE)*((-1/2)*stop_power_g))

    m00 = M[0, 0]
    m01 = M[0, 1]
    m10 = M[1, 0]
    m11 = M[1, 1]

    for i in range(0, I):
                                        # "left, right",  "down, up"
                                        # x then E
        ind_L_d = np.ravel_multi_index((i, 0, 0), (I, 2, 2))
        ind_L_u = np.ravel_multi_index((i, 0, 1), (I, 2, 2))
        ind_R_d = np.ravel_multi_index((i, 1, 0), (I, 2, 2))
        ind_R_u = np.ravel_multi_index((i, 1, 1), (I, 2, 2))

        # upwinding condition
        if (mu > 0):
            if (i > 0):
                ind_L_d_b = np.ravel_multi_index((i-1, 1, 0), (I, 2, 2))
                ind_L_u_b = np.ravel_multi_index((i-1, 1, 1), (I, 2, 2))
            ind_R_d_b = np.ravel_multi_index((i, 1, 0), (I, 2, 2))
            ind_R_u_b = np.ravel_multi_index((i, 1, 1), (I, 2, 2))
        else:
            ind_L_d_b = np.ravel_multi_index((i, 0, 0), (I, 2, 2))
            ind_L_u_b = np.ravel_multi_index((i, 0, 1), (I, 2, 2))
            if (i < I-1):
                ind_R_d_b = np.ravel_multi_index((i+1, 0, 0), (I, 2, 2))
                ind_R_u_b = np.ravel_multi_index((i+1, 0, 1), (I, 2, 2))


    # streaming
        mdx = mu*dx[i]
        # up, L
        if (i > 0) or (mu < 0):
            A.append((-1/6)*mdx, ind_L_u, ind_L_d_b)
            A.append((-1/3)*mdx, ind_L_u, ind_L_u_b)
        A.append((1/12)*mdx, ind_L_u, ind_L_d)
        A.append((1/12)*mdx, ind_L_u, ind_R_d)
        A.append((1/6)*mdx, ind_L_u, ind_L_u)
        A.append((1/6)*mdx, ind_L_u, ind_R_u)
        # up, R
        if (i<I-1) or (mu > 0):
            A.append((1/6)*mdx, ind_R_u, ind_R_d_b)
            A.append((1/3)*mdx, ind_R_u, ind_R_u_b)
        A.append((-1/12)*mdx, ind_R_u, ind_L_d)
        A.append((-1/12)*mdx, ind_R_u, ind_R_d)
        A.append((-1/6)*mdx, ind_R_u, ind_L_u)
        A.append((-1/6)*mdx, ind_R_u, ind_R_u)
        # dwn L
        if (i > 0) or (mu < 0):
            A.append((-1/3)*mdx, ind_L_d, ind_L_d_b)
            A.append((-1/6)*mdx, ind_L_d, ind_L_u_b)
        A.append((1/6)*mdx, ind_L_d, ind_L_d)
        A.append((1/6)*mdx, ind_L_d, ind_R_d)
        A.append((1/12)*mdx, ind_L_d, ind_L_u)
        A.append((1/12)*mdx, ind_L_d, ind_R_u)
        # dwn R
        if (i<I-1) or (mu > 0):
            A.append((1/3)*mdx, ind_R_d, ind_R_d_b)
            A.append((1/6)*mdx, ind_R_d, ind_R_u_b)
        A.append((-1/6)*mdx, ind_R_d, ind_L_d)
        A.append((-1/6)*mdx, ind_R_d, ind_R_d)
        A.append((-1/12)*mdx, ind_R_d, ind_L_u)
        A.append((-1/12)*mdx, ind_R_d, ind_R_u)
    
    # absorption + slowing-down
        # up, L
        A.append((v1[i]*m00),ind_L_u , ind_L_d)
        A.append((v1[i]*m01),ind_L_u , ind_R_d)
        A.append((v2[i]*m00),ind_L_u , ind_L_u)
        A.append((v2[i]*m01),ind_L_u , ind_R_u)
        # up, R
        A.append((v1[i]*m10),ind_R_u , ind_L_d)
        A.append((v1[i]*m11),ind_R_u , ind_R_d)
        A.append((v2[i]*m10),ind_R_u , ind_L_u)
        A.append((v2[i]*m11),ind_R_u , ind_R_u)
        
        # dwn L
        A.append((v3[i]*m00),ind_L_d , ind_L_d)
        A.append((v3[i]*m01),ind_L_d , ind_R_d)
        A.append((v4[i]*m00),ind_L_d , ind_L_u)
        A.append((v4[i]*m01),ind_L_d , ind_R_u)
        # dwn R
        A.append((v3[i]*m10),ind_R_d , ind_L_d)
        A.append((v3[i]*m11),ind_R_d , ind_R_d)
        A.append((v4[i]*m10),ind_R_d , ind_L_u)
        A.append((v4[i]*m11),ind_R_d , ind_R_u)

def form_HO_RHS(b, dx, dE, I, M, stop_power_bound, upwind_e_flux, scatter_source, q_g):
    # left boundary
    # right boundary
    m00 = M[0, 0]
    m01 = M[0, 1]
    m10 = M[1, 0]
    m11 = M[1, 1]


    for i in range(0, I):
        dxi = dx[i]
        ind_L_d = np.ravel_multi_index((i, 0, 0), (I, 2, 2))
        ind_L_u = np.ravel_multi_index((i, 0, 1), (I, 2, 2))
        ind_R_d = np.ravel_multi_index((i, 1, 0), (I, 2, 2))
        ind_R_u = np.ravel_multi_index((i, 1, 1), (I, 2, 2))
    # scattering source
        b.append((scatter_source[ind_L_u]), ind_L_u)
        b.append((scatter_source[ind_L_d]), ind_L_d)
        b.append((scatter_source[ind_R_d]), ind_R_d)
        b.append((scatter_source[ind_R_u]), ind_R_u)
    
    # external source
        b.append((dxi*m00/6)*(q_g[ind_L_d] + 2*q_g[ind_L_u]),ind_L_u)
        b.append((dxi*m01/6)*(q_g[ind_R_d] + 2*q_g[ind_R_u]),ind_L_u)
        b.append((dxi*m10/6)*(q_g[ind_L_d] + 2*q_g[ind_L_u]),ind_R_u)
        b.append((dxi*m11/6)*(q_g[ind_R_d] + 2*q_g[ind_R_u]),ind_R_u)

        b.append((dxi*m00/6)*(2*q_g[ind_L_d] + q_g[ind_L_u]),ind_L_d)
        b.append((dxi*m01/6)*(2*q_g[ind_R_d] + q_g[ind_R_u]),ind_L_d)
        b.append((dxi*m10/6)*(2*q_g[ind_L_d] + q_g[ind_L_u]),ind_R_d)
        b.append((dxi*m11/6)*(2*q_g[ind_R_d] + q_g[ind_R_u]),ind_R_d)
    # CSD source
        b.append(((dxi/dE)*stop_power_bound[i]*(m00*upwind_e_flux[ind_L_d] + m01*upwind_e_flux[ind_R_d])),ind_L_u)
        b.append(((dxi/dE)*stop_power_bound[i]*(m10*upwind_e_flux[ind_L_d] + m11*upwind_e_flux[ind_R_d])), ind_R_u)

    # Boundary cells TODO



def solve_DO(m : int,
             g : int,
             mesh : Mesh,
             xs_total_g,
             stop_power_g,
             stop_power_bound_down,
             stop_power_bound_up,
             scatter_source,
             upwind_e_flux,
             q_g
             ):
    # spatial mesh sweep for one Discrete Ordinate (DO) and energy group
    
    mu = mesh.mu[m]
    dE = mesh.dE[g]
    dx = mesh.dx
    I = mesh.I

    # Initialize Bilinear and Linear forms
    fespace = LD_space(mesh, g)
    A = bilinear(fespace)
    b = linear(fespace)

    size = 4*mesh.I

    

    b.zero()

    form_HO_LHS(A, mu, dx, dE, I, xs_total_g, stop_power_g, stop_power_bound_down, fespace.M)
    form_HO_RHS(b, dx, dE, I, fespace.M, stop_power_bound_up, upwind_e_flux, scatter_source, q_g)
    sA_iLU = scipy.sparse.linalg.spilu(A.matrix.tocsr())
    M = scipy.sparse.linalg.LinearOperator((size, size), sA_iLU.solve)
    # Solve Linear system
    x, _ = scipy.sparse.linalg.bicgstab(A.matrix.tocsr(), b.vector, M=M)
    # x = scipy.sparse.linalg.spsolve(A.matrix.tocsr(), b.vector)
    
    residual = 0
    if options.residuals == True:
        residual = (A.matrix.tocsr() @ x) - b.vector


    return x, residual

def sweep(g : int,
          mesh : Mesh,
          xs_total_g,
          stop_power_g,
          stop_power_bound_down,
          stop_power_bound_up,
          scatter_source,
          upwind_e_flux,
          q_g):
    # loop over all angles in a single energy group
    # return angular and scalar flux in group g

    print(f"performing sweep in group {g}")

    angular_flux = np.zeros((mesh.M, 4*mesh.I))
    
    psi = np.zeros((4*mesh.I))
    scalar_flux = np.zeros((4*mesh.I))

    for m in  range(0, mesh.M):
        psi, ang_residuals = solve_DO(m, g, mesh, xs_total_g, stop_power_g, 
                 stop_power_bound_down, stop_power_bound_up, scatter_source, upwind_e_flux[m], q_g)
        scalar_flux += mesh.w[m]*psi
        angular_flux[m] = psi
        if options.residuals == True:
            coord_max = np.unravel_index(np.argmax(ang_residuals), (mesh.I, 2, 2))
            coord_min = np.unravel_index(np.argmin(ang_residuals), (mesh.I, 2, 2))
            with open(options.residual_file, "a") as txt:
                txt.write(f"g = {g},    m={m}\n")
                txt.write(f"maximum = {np.max(ang_residuals)} at {np.argmax(ang_residuals)} (i={coord_max[0]}, L/R={coord_max[1]}, beta={coord_max[2]})\n")
                txt.write(f"minimum = {np.min(ang_residuals)} at {np.argmin(ang_residuals)} (i={coord_min[0]}, L/R={coord_min[1]}, beta={coord_min[2]})\n")
                txt.write("\n")
        
    return scalar_flux, angular_flux

def compute_scatter_source(g, scalar_g, scalar, xs_scatter, mesh : Mesh):
    I = mesh.I
    dx = mesh.dx
    dE = mesh.dE
    source = np.zeros((4*mesh.I))

    fespace = LD_space(mesh, g)
    m00 = fespace.M[0, 0]
    m01 = fespace.M[0, 1]
    m10 = fespace.M[1, 0]
    m11 = fespace.M[1, 1]
    
    
    for gp in range(0, mesh.G):
        sc = scalar[gp]
         
        if gp == g:
            sc = scalar_g


        for i in range(0, I):
            ind_L_d = np.ravel_multi_index((i, 0, 0), (I, 2, 2))
            ind_L_u = np.ravel_multi_index((i, 0, 1), (I, 2, 2))
            ind_R_d = np.ravel_multi_index((i, 1, 0), (I, 2, 2))
            ind_R_u = np.ravel_multi_index((i, 1, 1), (I, 2, 2))

            source[ind_L_u] += (dx[i]*dE[gp]/8)*xs_scatter[gp, g, i]*((m00*(sc[ind_L_u]+sc[ind_L_d]))+(m01*(sc[ind_R_u]+sc[ind_R_d])))
            source[ind_R_u] += (dx[i]*dE[gp]/8)*xs_scatter[gp, g, i]*((m10*(sc[ind_L_u]+sc[ind_L_d]))+(m11*(sc[ind_R_u]+sc[ind_R_d])))

            source[ind_L_d] += (dx[i]*dE[gp]/8)*xs_scatter[gp, g, i]*((m00*(sc[ind_L_u]+sc[ind_L_d]))+(m01*(sc[ind_R_u]+sc[ind_R_d])))
            source[ind_R_d] += (dx[i]*dE[gp]/8)*xs_scatter[gp, g, i]*((m10*(sc[ind_L_u]+sc[ind_L_d]))+(m11*(sc[ind_R_u]+sc[ind_R_d])))

    return source




def high_order_ingroup_iteration(g: int,
                         mesh : Mesh,
                         initial_angular,
                         scalar,
                         xs_total_g,
                         stop_power_g,
                         stop_power_bound_down,
                         stop_power_bound_up,
                         upwind_e_flux,
                         group_scalar,
                         xs_scatter,
                         q_g):
    # iterate on within-group scattering source ("inner iteration")
    # call sweep() until convergence
    # return angular flux distribution in group g

    rel_change = 1
    s = 0

    # initial_angular = np.ones((mesh.M, 4*mesh.I))
    initial_scalar = np.transpose(initial_angular) @ mesh.w
    scalar_g = initial_scalar.copy()
    angular = initial_angular.copy()

    while (rel_change > options.epsilon["inner"]):
        s+=1
        scatter_source = compute_scatter_source(g, scalar_g, scalar, xs_scatter, mesh)

        prev_angular = angular*1
        
        scalar_g, angular = sweep(g, mesh, xs_total_g, stop_power_g, stop_power_bound_down, stop_power_bound_up, scatter_source, upwind_e_flux, q_g)
        max_change = np.max(np.abs(prev_angular - angular)*np.power(prev_angular, -1))
        max_loc = np.argmax(np.abs(prev_angular - angular)*np.power(prev_angular, -1))
        rel_change = np.linalg.norm(np.abs(prev_angular - angular)/prev_angular)
        print(f"s: L2 {rel_change}, max {max_change} at {max_loc}")

    return scalar_g, angular

def calculate_closure(F_closure, F_bound, angular_flux, mesh : Mesh):
    # calculate 'F' closure term for a single group for use in SMM equation.
    # F = int_{-1}^1 d\mu ((1/3) - \mu&2)\psi(x, \mu)

    # Result : write F and F_bound (cell-edge closure) to variables 'F_closure' and 'F_bound'


    pass

def energy_pass(mesh : Mesh):

    # within each energy group, iterate on scattering source

    scalar_result = np.zeros((mesh.G, 4*mesh.I))
    angular_result = np.zeros((mesh.G, mesh.M, 4*mesh.I))
    initial_angular = np.zeros((mesh.G, mesh.M, 4*mesh.I))

    upwind_e_flux = np.zeros((mesh.M, 4*mesh.I))

    for g in range(0, mesh.G):

        if g > 0:
            stop_power_bound_up = mesh.stopping_power_d[g-1]
        else:
            stop_power_bound_up = 0 * (mesh.stopping_power_d[g-1])
            upwind_e_flux = np.zeros((mesh.M, 4*mesh.I))

        change = 1
        n=0
        while (n < 1):
        # while (change > options.epsilon["outer"]):
            n += 1
            print(f"loop {n} of group {g}")
            scalar, angular = high_order_ingroup_iteration(g,
                            mesh,
                            angular_result[g],
                            scalar_result,
                            mesh.xs_total[g],
                            mesh.stopping_power[g],
                            mesh.stopping_power_d[g],
                            stop_power_bound_up,
                            upwind_e_flux,
                            scalar_result[g],
                            mesh.xs_scatter,
                            mesh.angular_source[g])
            change = np.linalg.norm(np.abs((scalar_result[g] - scalar)/scalar_result[g]))
            scalar_result[g] = scalar
            angular_result[g] = angular

        # upwind_e_flux =  angular
    return scalar_result, angular

    # return angualar flux distribution for full energy spectrum

# def iterate_groups
    # iterate on upscattering ("outer iteration")
    # loop energy_pass() until flux is converged
    # return angular flux distribution for full energy spectrum