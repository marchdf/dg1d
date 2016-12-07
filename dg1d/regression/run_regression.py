#!/usr/bin/env python
#
# Run the regression tests
#
#

#================================================================================
#
# Imports
#
#================================================================================
import os
import sys
import shutil
import subprocess as sp

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
#
# Problem setup
#
#================================================================================
basedir = os.getcwd()
codedir = os.path.dirname(basedir)
datadir = basedir

#================================================================================
# Sod shock tube
workdir = datadir+'/sodtube'
os.chdir(workdir)
runcode('deck.inp',codedir,False)

os.chdir(basedir)

