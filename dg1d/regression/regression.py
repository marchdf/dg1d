#================================================================================
#
# Imports
#
#================================================================================
import os
import sys
import glob
import subprocess as sp
import unittest
import numpy as np
import numpy.testing as npt


#================================================================================
#
# Function definitions
#
#================================================================================
def runcode(workdir,deck,codedir,background=True):
    """Run the DG code (in background by default) given the input deck

    If background is False, then wait for the process to finish and
    give me a return code.
    """

    cwd = os.getcwd()
    os.chdir(workdir)
    
    # Launch the code
    log = open('logfile', "w")
    proc = sp.Popen(codedir+'/main.py -d '+deck,
                    shell=True, stdout=log,stderr=sp.PIPE)
    retcode = 0

    # If you don't want to send the process to the background
    if (not background):
        retcode = proc.wait()
        log.flush()

    os.chdir(cwd)
    return retcode

#================================================================================
def compare_with_golds(workdir):
    """Compare new solution to the gold results"""

    # Test density
    dat  = np.loadtxt(os.path.join(workdir,'rho0000000001.dat'),delimiter=',')
    gold = np.loadtxt(os.path.join(workdir,'rho0000000001.gold'),delimiter=',')
    npt.assert_array_almost_equal(dat,gold,decimal=7,
                                  err_msg='Failed on density comparison')
    
    # Test momentum
    dat  = np.loadtxt(os.path.join(workdir,'rhou0000000001.dat'),delimiter=',')
    gold = np.loadtxt(os.path.join(workdir,'rhou0000000001.gold'),delimiter=',')
    npt.assert_array_almost_equal(dat,gold,decimal=7,
                                  err_msg='Failed on momentum comparison')

    # Test energy
    dat  = np.loadtxt(os.path.join(workdir,'E0000000001.dat'),delimiter=',')
    gold = np.loadtxt(os.path.join(workdir,'E0000000001.gold'),delimiter=',')
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
    # Execute a test
    def launch(self,workdir):
        """Execute the sequence of command to run a test"""
        
        # Run code
        [os.remove(f) for f in glob.glob(os.path.join(workdir,'*.dat'))]
        runcode(workdir,'deck.inp',self.codedir,False)
        
        # Test with gold
        compare_with_golds(workdir)
        
        # If test passed, make a plot
        self.runplot(workdir,False)

        
    #================================================================================
    # Makes the plots
    def runplot(self,workdir,background=True):
        """Run the plot tool on a problem directory
        
        If background is False, then wait for the process to finish and
        give me a return code.
        """

        # Launch the plot
        proc = sp.Popen(self.regdir+'/plot.py -f 1 -d '+workdir,
                        shell=True, stdout=sp.PIPE,stderr=sp.PIPE)
        retcode = 0
        
        # If you don't want to send the process to the background
        if (not background):
            retcode = proc.wait()

        return retcode


    #================================================================================
    # Sod shock tube
    def test_sodtube(self):
        """Is the Sod shock tube problem correct?"""
        workdir = self.regdir+'/sodtube'
        self.launch(workdir)


    #================================================================================
    # modified Sod shock tube
    def test_modified_sodtube(self):
        """Is the modified Sod shock tube problem correct?"""
        workdir = self.regdir+'/sodtube_modified'
        self.launch(workdir)


    #================================================================================
    # 123 problem
    def test_123_problem(self):
        """Is the 123 problem correct?"""
        workdir = self.regdir+'/123_problem'
        self.launch(workdir)
        

    #================================================================================
    # Left Woodward and Colella (blast wave)
    def test_blast_wave(self):
        """Is the blast_wave problem correct?"""
        workdir = self.regdir+'/blast_wave'
        self.launch(workdir)
        

    #================================================================================
    # collision of two strong shocks
    def test_strong_shocks(self):
        """Is the collision of two strong shocks problem correct?"""
        workdir = self.regdir+'/strong_shocks'
        self.launch(workdir)
        

    #================================================================================
    # stationary contact discontinuity
    def test_stationary_contact(self):
        """Is the stationary contact discontinuity problem correct?"""
        workdir = self.regdir+'/stationary_contact'
        self.launch(workdir)


if __name__ == '__main__':
    unittest.main()
