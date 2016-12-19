#================================================================================
#
# Imports
#
#================================================================================
import os
import sys
import glob
import shutil
import subprocess as sp
import argparse
import unittest
import numpy as np
import numpy.testing as npt


#================================================================================
#
# Function definitions
#
#================================================================================
def runcode(deck,codedir,background=True):
    """Run the DG code (in background by default) given the input deck

    If background is False, then wait for the process to finish and
    give me a return code.
    """

    # Launch the code
    log = open('logfile', "w")
    proc = sp.Popen(codedir+'/main.py -d '+deck, shell=True, stdout=log,stderr=sp.PIPE)
    retcode = 0

    # If you don't want to send the process to the background
    if (not background):
        retcode = proc.wait()
        log.flush()

    return retcode

#================================================================================
def runplot(workdir,background=True):
    """Run the plot tool on a problem directory

    If background is False, then wait for the process to finish and
    give me a return code.
    """

    # Launch the plot
    proc = sp.Popen('./plot.py -f 1 -d '+workdir, shell=True, stdout=sp.PIPE,stderr=sp.PIPE)
    retcode = 0

    # If you don't want to send the process to the background
    if (not background):
        retcode = proc.wait()

    return retcode


#================================================================================
def compare_with_golds(workdir):
    """Compare new solution to the gold results"""

    # Test density
    dat  = np.loadtxt('rho0000000001.dat',delimiter=',')
    gold = np.loadtxt('rho0000000001.gold',delimiter=',')
    npt.assert_array_almost_equal(dat,gold,decimal=7,
                                  err_msg='Failed on density comparison')
    
    # Test momentum
    dat  = np.loadtxt('rhou0000000001.dat',delimiter=',')
    gold = np.loadtxt('rhou0000000001.gold',delimiter=',')
    npt.assert_array_almost_equal(dat,gold,decimal=7,
                                  err_msg='Failed on momentum comparison')

    # Test energy
    dat  = np.loadtxt('E0000000001.dat',delimiter=',')
    gold = np.loadtxt('E0000000001.gold',delimiter=',')
    npt.assert_array_almost_equal(dat,gold,decimal=7,
                                  err_msg='Failed on energy comparison')
        

#================================================================================
#
# Class definitions
#
#================================================================================
class RegressionTestCase(unittest.TestCase):
    """Regression tests for `main.py.`"""


    #================================================================================
    # Set up
    def setUp(self):

        # Problem setup
        self.codedir = os.getcwd() 
        self.regdir  = os.path.join(self.codedir,'regression')

        
    #================================================================================
    # Sod shock tube
    def test_sodtube(self):
        """Is the Sod shock tube problem correct?"""

        workdir = self.regdir+'/sodtube'
        os.chdir(workdir)
        [os.remove(fname) for fname in glob.glob("*.dat")]
        
        # Run code
        runcode('deck.inp',self.codedir,False)

        # Test with gold
        compare_with_golds(workdir)

        # If test passed, make a plot
        os.chdir(self.regdir)
        runplot(workdir,False)

        
