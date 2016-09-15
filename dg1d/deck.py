#================================================================================
#
# Imports
#
#================================================================================

#================================================================================
#
# Class definitions
#
#================================================================================
class Deck:
    'Given a deck file, create a deck object'

    #================================================================================
    def __init__(self):

        # Typical values for the deck
        self.system = "advection"
        self.rk = ''
        self.ic = "sinewave"
        self.nout = 10
        self.finaltime = 1
        self.cfl = 0.5
        self.order = 1
        self.limiting = ''
        self.enhance = ''
        self.sensor_thresholds = []

    #================================================================================
    def parser(self,fname):
        """Parse the deck file

        """
        print("Parsing the deck file.")
        with open(fname) as f:
            for line in f:
                if "#PDE system" in line:
                    self.system = next(f).rstrip()
                elif "#RK scheme" in line:
                    self.rk = next(f).rstrip()
                elif "#initial condition" in line:
                    self.ic = next(f).rstrip()
                elif "#number of outputs" in line:
                    self.nout = int(next(f))
                elif "#final time" in line:
                    self.finaltime = float(next(f))
                elif "#Courant-Friedrichs-Lewy condition" in line:
                    self.cfl = float(next(f))
                elif "#order" in line:
                    self.order = int(next(f))
                elif "#limiting" in line:
                    self.limiting = next(f).rstrip()
                elif "#enhancement" in line:
                    self.enhance = next(f).rstrip()
                elif "#sensor thresholds" in line:
                    line = next(f).rstrip()
                    self.sensor_thresholds = [float(i) for i in line.split()]

                
#================================================================================
def write_deck(WORKDIR,defs):
    """Writes a deck file to WORKDIR given some problem definitions
    
    Returns the filename of the deck
    """
    
    deckname = WORKDIR+'/deck.inp'
    
    with open(deckname, "w") as f:
        for l in defs:
            f.write('#'+l[0]+'\n'+l[1]+'\n')
            
    return deckname 

