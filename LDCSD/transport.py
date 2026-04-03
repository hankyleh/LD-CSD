from .fem import *
from .mesh import Mesh
import numpy as np
import scipy

from LDCSD import options

def form_HO_LHS(A, mu, dx, dE, I, xs_total_g, stop_power_g, stop_power_bound, M):
    
    # coefficients for absorption + streaming equations
    v1 = dx*((1/6)*(xs_total_g) + (1/(2*dE))*(stop_power_g))
    v2 = dx*((1/3)*(xs_total_g) + (1/(2*dE))*(stop_power_g))
    v3 = dx*((1/6)*(xs_total_g) + (1/dE)*((-1/2)*stop_power_g + stop_power_bound))
    v4 = dx*((1/3)*(xs_total_g) + (1/dE)*((-1/2)*stop_power_g))

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
        b.append((dxi*m00/6)*(q_g[0,0,i] + 2*q_g[0,1,i]),ind_L_u)
        b.append((dxi*m01/6)*(q_g[1,0,i] + 2*q_g[1,1,i]),ind_L_u)
        b.append((dxi*m10/6)*(q_g[0,0,i] + 2*q_g[0,1,i]),ind_R_u)
        b.append((dxi*m11/6)*(q_g[1,0,i] + 2*q_g[1,1,i]),ind_R_u)

        b.append((dxi*m00/6)*(2*q_g[0,0,i] + q_g[0,1,i]),ind_L_d)
        b.append((dxi*m01/6)*(2*q_g[1,0,i] + q_g[1,1,i]),ind_L_d)
        b.append((dxi*m10/6)*(2*q_g[0,0,i] + q_g[0,1,i]),ind_R_d)
        b.append((dxi*m11/6)*(2*q_g[1,0,i] + q_g[1,1,i]),ind_R_d)
    # CSD source
        b.append(((dxi/dE)*stop_power_bound[i]*(m00*upwind_e_flux[0] + m01*upwind_e_flux[1])),ind_L_u)
        b.append(((dxi/dE)*stop_power_bound[i]*(m10*upwind_e_flux[0] + m11*upwind_e_flux[1])), ind_R_u)
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


    form_HO_LHS(A, mu, dx, dE, I, xs_total_g, stop_power_g, stop_power_bound_down, fespace.M)
    form_HO_RHS(b, dx, dE, I, fespace.M, stop_power_bound_up, upwind_e_flux, scatter_source, q_g)


    # Solve Linear system
    x = scipy.sparse.linalg.spsolve(A.matrix.tocsr(), b.vector)
    
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

    angular_flux = numpy.zeros((mesh.M, 4*mesh.I))
    scalar_flux = numpy.zeros((4*mesh.I))
    psi = numpy.zeros((4*mesh.I))
    
    for m in  range(0, mesh.M):
        psi, ang_residuals = solve_DO(m, g, mesh, xs_total_g, stop_power_g, 
                 stop_power_bound_down, stop_power_bound_up, scatter_source, upwind_e_flux, q_g)
        scalar_flux += mesh.w[m]*psi*0.5
        angular_flux[m] = psi
        if options.residuals == True:
            coord_max = np.unravel_index(np.argmax(ang_residuals), (mesh.I, 2, 2))
            coord_min = np.unravel_index(np.argmin(ang_residuals), (mesh.I, 2, 2))
            with open(options.residual_file, "a") as txt:
                txt.write(f"g = {g},    m={m}\n")
                txt.write(f"maximum = {np.max(ang_residuals)} at {np.argmax(ang_residuals)} (i={coord_max[0]}, L/R={coord_max[1]}, beta={coord_max[2]})\n")
                txt.write(f"minimum = {np.min(ang_residuals)} at {np.argmin(ang_residuals)} (i={coord_min[0]}, L/R={coord_min[1]}, beta={coord_min[2]})\n")
                txt.write(f"L2 = {numpy.linalg.norm(ang_residuals)}\n")
                txt.write("\n")
        
    return scalar_flux, angular_flux

def compute_scatter_source(g, scalar_g, scalar, xs_scatter, mesh : Mesh):
    I = mesh.I
    dx = mesh.dx
    dE = mesh.dE
    source = numpy.zeros((4*mesh.I))

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

    initial_angular = numpy.ones((mesh.M, 4*mesh.I))
    initial_scalar = numpy.transpose(initial_angular) @ mesh.w
    # print(initial_scalar)

    angular = initial_angular.copy()
    scalar = 0.5*numpy.transpose(initial_angular) @ mesh.w

    while (rel_change > options.epsilon["inner"]):
        s+=1
        scatter_source = compute_scatter_source(g, scalar, group_scalar, xs_scatter, mesh)

        prev_angular = angular*1
        
        scalar, angular = sweep(g, mesh, xs_total_g, stop_power_g, stop_power_bound_down, stop_power_bound_up, scatter_source, upwind_e_flux, q_g)
        # print(scalar)
        # print(scatter_source)
        # print(q_g)
        rel_change = np.linalg.norm(np.abs(prev_angular - angular)*np.power(prev_angular, -1))
        print(f"s: {rel_change}")

    return scalar, angular

        # compute scattering source from initial guess

        # call sweep with this scattering source
        



# def energy_pass():

    # return angualar flux distribution for full energy spectrum

# def iterate_groups
    # iterate on upscattering ("outer iteration")
    # loop energy_pass() until flux is converged
    # return angular flux distribution for full energy spectrum