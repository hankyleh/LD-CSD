import numpy as np

class Mesh:
    def __init__(self, x_bounds, group_bounds, regions, boundaries, angles, weights):
        self.x_bounds = x_bounds
        self.group_bounds = group_bounds
        self.G = group_bounds.size-1
        self.I = x_bounds.size-1
        self.regions = regions
        self.boundaries = boundaries
        self.mu = angles
        self.w = weights
        self.cell_centers = (x_bounds[1:]+x_bounds[0:-1]) * 0.5
        self.dx = (x_bounds[1:] - x_bounds[0:-1])
        self.dE = (group_bounds[1:] - group_bounds[0:-1])

        if weights.size != angles.size:
            raise Exception(f"Attempted to initialize mesh with {angles.size} angles and {weights.size} weights")
        self.M = angles.size

        self.xs_total = np.zeros((self.G, self.I))
        self.xs_scatter = np.zeros((self.G,self.G, self.I))
        self.stopping_power = np.zeros((self.G, self.I))
        self.scalar_source = np.zeros((self.G, self.I))
        self.angular_source = np.zeros((self.G, self.I))

        for r in range(len(regions.materials)):
            truth = 1*(self.cell_centers < regions.bounds[r+1])*(self.cell_centers > regions.bounds[r])
            index = np.trim_zeros(np.array(truth * (np.arange(0, self.I)+1))) - 1
            for g in range(0, self.G):
                self.xs_total[g, index] = regions.materials[r].total[g]
                self.stopping_power[g, index] = regions.materials[r].stopping_power[g]
                self.scalar_source[g, index] = regions.materials[r].scalar_source[g]
                self.angular_source[g, index] = regions.materials[r].angular_source[g]
                for gp in range(0, self.G):
                    self.xs_scatter[g, gp,index] = regions.materials[r].scatter[g, gp]


