#================================================================================
#
# Imports
#
#================================================================================
import os

#================================================================================
#
# Function definitions
#
#================================================================================

#================================================================================
def makedir(directory):
    """Makes a directory if it does not exist.
    
    Taken from http://stackoverflow.com/questions/273192/how-to-check-if-a-directory-exists-and-create-it-if-necessary
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
