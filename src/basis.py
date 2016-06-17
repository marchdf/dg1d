
class Basis:
    'Generate the basis functions, gradients and Gaussian quadrature'
    
    def __init__(self,order):

        self.order = order
        
    def legendre(self,fname):
        """Returns Legendre polynomial coefficients

        """
        print("Parsing the deck file")
        with open(fname) as f:
            for line in f:
                if "#PDE system" in line:
                    self.system = next(f)
                elif "#initial condition" in line:
                    self.ic = next(f)
                elif "#output time step size" in line:
                    self.dtout = next(f)
                elif "#final time" in line:
                    self.finaltime = next(f)
                elif "#Courant-Friedrichs-Lewy condition" in line:
                    self.cfl = next(f)
                elif "#order" in line:
                    self.order = next(f)
                elif "#limiting" in line:
                    self.limiting = next(f)

