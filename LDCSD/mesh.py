import numpy as np
import LDCSD

class Mesh:
    def __init__(self, x_edges, group_boundaries, mat_regions, angles, weights):
        global mu, w, x_bounds, I, regions, boundaries, cell_centers, dx, M, group_bounds, G, dE
        LDCSD.x_bounds = x_edges

        LDCSD.I = x_edges.size-1
        LDCSD.regions = mat_regions
        LDCSD.cell_centers = (x_edges[1:]+x_edges[0:-1]) * 0.5
        LDCSD.dx = (x_edges[1:] - x_edges[0:-1])


        self.xs_total = np.zeros((LDCSD.G, LDCSD.I))
        self.xs_scatter = np.zeros((LDCSD.G,LDCSD.G, LDCSD.I))
        self.stopping_power = np.zeros((LDCSD.G, LDCSD.I))
        self.stopping_power_d = np.zeros((LDCSD.G, LDCSD.I))
        self.scalar_source = np.zeros((LDCSD.G, 4*LDCSD.I))
        self.angular_source = np.zeros((LDCSD.G, LDCSD.M, 4*LDCSD.I))

        for r in range(len(mat_regions.materials)):
            truth = 1*(LDCSD.cell_centers <= mat_regions.bounds[r+1])*(LDCSD.cell_centers > mat_regions.bounds[r])
            index = np.trim_zeros(np.array(truth * (np.arange(0, LDCSD.I)+1))) - 1
            for g in range(0, LDCSD.G):
                self.xs_total[g, index] = mat_regions.materials[r].total[g]
                self.stopping_power[g, index] = mat_regions.materials[r].stopping_power[g]
                self.stopping_power_d[g, index] = mat_regions.materials[r].stopping_power_d[g]
                
                ld = np.ravel_multi_index((index, 0, 0), (LDCSD.I, 2, 2))
                lu = np.ravel_multi_index((index, 0, 1), (LDCSD.I, 2, 2))
                rd = np.ravel_multi_index((index, 1, 0), (LDCSD.I, 2, 2))
                ru = np.ravel_multi_index((index, 1, 1), (LDCSD.I, 2, 2))

                self.scalar_source[g, ld] = mat_regions.materials[r].scalar_source[g]
                self.scalar_source[g, lu] = mat_regions.materials[r].scalar_source[g]
                self.scalar_source[g, rd] = mat_regions.materials[r].scalar_source[g]
                self.scalar_source[g, ru] = mat_regions.materials[r].scalar_source[g]

                self.angular_source[g,:, ld] = mat_regions.materials[r].angular_source[g]
                self.angular_source[g,:, lu] = mat_regions.materials[r].angular_source[g]
                self.angular_source[g,:, rd] = mat_regions.materials[r].angular_source[g]
                self.angular_source[g, :,ru] = mat_regions.materials[r].angular_source[g]

                for gp in range(0, LDCSD.G):
                    self.xs_scatter[g, gp,index] = mat_regions.materials[r].scatter[g, gp]


