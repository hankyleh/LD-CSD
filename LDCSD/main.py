from LDCSD import mesh, regions, boundaries, transport


def run(mesh, mode="high_order"):
    if mode == "high_order":
        # run high-order iteration scheme here
        
        # initialize FE parameters

        # outer iteration
            # inner iteration
                # sweep for each angle
                
        transport.solve_DO(mesh)
        return 0
    else:
        raise Exception(f"Unrecognized run mode '{mode}'")
        return 1
    
