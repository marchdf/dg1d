#================================================================================
#
# Imports
#
#================================================================================
import numpy as np

import solution

#================================================================================
#
# Class definitions
#
#================================================================================
class Sensor:
    'Determines the sensors for limiting'

    #================================================================================
    def __init__(self,thresholds,solution):
        
        print("Setting up the sensors.")

        # the sensors
        self.sensors = np.zeros((solution.N_E+2),dtype = int)

        # the criteria
        self.thresholds = thresholds

    #================================================================================
    def sensing(self,solution):
        """Find where limiting needs to be done"""

        # Ensure that the sensors are clean
        self.sensors.fill(0)

        # Calculate the sensors
        solution.keywords['sensing'](self.sensors,self.thresholds,solution)
