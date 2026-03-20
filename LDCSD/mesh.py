class Mesh:
    def __init__(self, x_bounds, group_bounds, regions, boundaries):
        self.x_bounds = x_bounds
        self.group_bounds = group_bounds
        self.G = group_bounds.size-1
        self.I = x_bounds.size-1
        self.regions = regions
        self.boundaries = boundaries

        self.cell_centers = (x_bounds[1:]+x_bounds[0:-1]) * 0.5
