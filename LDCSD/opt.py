class Options:
    def __init__(self, **kwargs):
        # usage example:
        # options(print_sweep_residual = True, warning_print_level = 3)
        self.print_index = {"outer" : False, 
                            "inner" : False}
        self.scheme = {"method" : "high_order_transport", 
                       "lumped" : False, 
                       "matrix_solver" : "spsolve"}
        self.residuals = False
        self.keep_angular = False
        self.epsilon = {"inner" : 1e-5,
                        "outer" : 1e-6}
        
        
        if kwargs.get("silence", True):
            self.silence()
        if 'method' in kwargs:
            print(kwargs.get("method"))
            self.scheme["method"] = kwargs.get("method")

    def silence(self):
        # silence all outputs
        for key in self.print_index:
            self.print_index[key] = False
    def output_residuals(self):
        self.residual_file = "residuals.txt"
        with open("residuals.txt", "w") as txt:
            txt.write("")
        self.residuals = True
    def keep_angular(self):
        self.keep_angular = True