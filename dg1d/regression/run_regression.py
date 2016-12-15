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
import glob
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
map(os.remove, glob.glob("*.dat"))
runcode('deck.inp',codedir,False)
os.chdir(basedir)
runplot(workdir,False)

#================================================================================
# modified Sod shock tube
workdir = datadir+'/sodtube_modified'
os.chdir(workdir)
map(os.remove, glob.glob("*.dat"))
runcode('deck.inp',codedir,False)
os.chdir(basedir)
runplot(workdir,False)

#================================================================================
# 123 problem
workdir = datadir+'/123_problem'
os.chdir(workdir)
map(os.remove, glob.glob("*.dat"))
runcode('deck.inp',codedir,False)
os.chdir(basedir)
runplot(workdir,False)

#================================================================================
# Left Woodward and Colella (blast wave)
workdir = datadir+'/blast_wave'
os.chdir(workdir)
map(os.remove, glob.glob("*.dat"))
runcode('deck.inp',codedir,False)
os.chdir(basedir)
runplot(workdir,False)

#================================================================================
# collision of two strong shocks
workdir = datadir+'/strong_shocks'
os.chdir(workdir)
map(os.remove, glob.glob("*.dat"))
runcode('deck.inp',codedir,False)
os.chdir(basedir)
runplot(workdir,False)

#================================================================================
# stationary contact discontinuity
workdir = datadir+'/stationary_contact'
os.chdir(workdir)
map(os.remove, glob.glob("*.dat"))
runcode('deck.inp',codedir,False)
os.chdir(basedir)
runplot(workdir,False)

