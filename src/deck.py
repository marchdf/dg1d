
class Deck:
    'Given a deck file, create a deck object'
    
    def __init__(self):

        # Typical values for the deck
        self.system = "advection"
        self.ic = "sinewave"
        self.nout = 10
        self.finaltime = 1
        self.cfl = 0.5
        self.order = 1
        self.limiting = False

        
    def parser(self,fname):
        """Parse the deck file

        """
        print("Parsing the deck file")
        with open(fname) as f:
            for line in f:
                if "#PDE system" in line:
                    self.system = next(f)
                elif "#initial condition" in line:
                    self.ic = next(f)
                elif "#number of outputs" in line:
                    self.nout = int(next(f))
                elif "#final time" in line:
                    self.finaltime = float(next(f))
                elif "#Courant-Friedrichs-Lewy condition" in line:
                    self.cfl = float(next(f))
                elif "#order" in line:
                    self.order = int(next(f))
                elif "#limiting" in line:
                    self.limiting = next(f)

                
