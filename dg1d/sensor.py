#=========================================================================
#
# Imports
#
#=========================================================================
import numpy as np

#=========================================================================
#
# Class definitions
#
#=========================================================================


class Sensor:
    'Determines the sensors for limiting'

    #=========================================================================
    def __init__(self, thresholds, length):

        print("Setting up the sensors.")

        # the criteria
        self.thresholds = thresholds

        # the sensors
        self.sensors = np.zeros((length), dtype=int)

    #=========================================================================
    def sensing(self, solution):
        """Find where limiting needs to be done"""

        # Ensure that the sensors are clean
        self.sensors.fill(0)

        # Calculate the sensors
        solution.keywords['sensing'](self.sensors, self.thresholds, solution)
