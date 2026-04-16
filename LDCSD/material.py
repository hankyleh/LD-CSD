class Material:
    def __init__(self,
                 G,
                 stopping_power=None, 
                 stopping_power_d=None,
                 total=None, 
                 scatter=None, 
                 scalar_source =None, 
                 angular_source=None 
                 ):
        self.stopping_power = stopping_power
        self.stopping_power_d = stopping_power_d
        self.total = total
        self.scatter = scatter
        self.scalar_source = scalar_source
        self.angular_source = angular_source

        if (self.scalar_source is not None) and (self.angular_source is None):
            self.angular_source = 0.5 * self.scalar_source
        elif (self.scalar_source is None) and (self.angular_source is not None):
            self.scalar_source = 2 * self.angular_source
        
        # TODO set 'None' values to Gx1 array of zeros