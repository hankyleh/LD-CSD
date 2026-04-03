from LDCSD import mesh, regions, boundaries, transport
from LDCSD import options
# from LDCSD.options import options
import numpy as np


def run(mesh):
    if options.scheme["method"] == "high_order_transport":
        # run high-order iteration scheme here
        
        # initialize FE parameters

        # outer iteration
            # inner iteration
                # sweep for each angle
        upwind_e_flux = np.array([0.001, 0.001])
        scatter_source = np.array([0, 0])
        q_g = np.zeros((2, 2, mesh.I))
        q_g = q_g+0.03
                
        # angular, scalar = transport.sweep(1,mesh, mesh.xs_total[0], mesh.stopping_power[1], mesh.stopping_power[0], mesh.stopping_power[0], np.array([0.001, 0.001]), upwind_e_flux, q_g)

        angular = transport.high_order_ingroup_iteration(1,
                         mesh,
                         0.001,
                         np.zeros((mesh.M, 4*mesh.I)),
                         mesh.xs_total[1],
                         mesh.stopping_power[1],
                         mesh.stopping_power[1],
                         mesh.stopping_power[0],
                         np.array([0.001, 0.001]),
                         np.zeros((mesh.G, 4*mesh.I))+0.1,
                         mesh.xs_scatter,
                         q_g)
        
        # print(f"size: {angular.size}")
        # print(f"scalar size: {scalar.size}")
        return angular
    else:
        raise Exception(f"Unrecognized run mode '{options.scheme["method"]}'")
        return 1
    
