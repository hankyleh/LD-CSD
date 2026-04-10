from LDCSD import mesh, regions, boundaries, transport
from LDCSD import options
# from LDCSD.options import options
import numpy as np


def high_order(mesh):
    if options.scheme["method"] == "high_order_transport":
        # run high-order iteration scheme here
        scalar, angular = transport.energy_pass(mesh)
        
        return scalar, angular
    else:
        raise Exception(f"Unrecognized run mode '{options.scheme["method"]}'")
        return 1
    
