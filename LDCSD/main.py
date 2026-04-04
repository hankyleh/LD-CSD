from LDCSD import mesh, regions, boundaries, transport
from LDCSD import options
# from LDCSD.options import options
import numpy as np


def run(mesh):
    if options.scheme["method"] == "high_order_transport":
        # run high-order iteration scheme here
        scalar, _ = transport.energy_pass(mesh)
        
        return scalar
    else:
        raise Exception(f"Unrecognized run mode '{options.scheme["method"]}'")
        return 1
    
