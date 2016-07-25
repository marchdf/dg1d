#!/usr/bin/env python
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

codedir = '/home/marchdf/dg1d/dg1d/'
sys.path.insert(0, codedir)
import deck as deck

#================================================================================
#
# Function definitions
#
#================================================================================
def runcode(deck,background=True):
    """Run the DG code (in background by default) given the input deck

    If background is False, then wait for the process to finish and
    give me a return code.
    """

    # Launch the code
    log = open('logfile', "w")
    proc = sp.Popen(codedir+'main.py -d '+deck, shell=True, stdout=log,stderr=sp.PIPE)
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
datadir = basedir

orders = [1]
resolutions = [4,8,16,32,64,128,256,512,1024]  #[8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768]

orders = [2]
resolutions = [4,8,16,32,64,128,256,512]  #[8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768]

orders = [3]
resolutions = [4,8,16,32,64,128]  #[8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768]

orders = [4]
resolutions = [4,8,16,32,64]  #[8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768]


for p, order in enumerate(orders):

    # remove old order directory
    shutil.rmtree(datadir + '/' + str(order), ignore_errors=True)

    for k,res in enumerate(resolutions):
    

        # problem definitions
        defs = [ ['PDE system' , 'advection'],
                 ['RK scheme' , 'rk12'],
                 ['initial condition' , 'sinewave '+str(res)],
                 ['number of outputs' , '11'],
                 ['final time', '8'],
                 ['Courant-Friedrichs-Lewy condition' , '0.5'],
                 ['order' , str(order)],
                 ['limiting' , '0']]


        # working directory for the data
        workdir =  datadir + '/' + str(order)+'/'+str(res)
        print('Creating directory',workdir,'and go to it')
        os.makedirs(workdir)
        os.chdir(workdir)
        
        # create the deck
        deckname = deck.write_deck(workdir,defs)
        
        # run the code
        runcode(deckname,False)
        
        # Go back to our base directory
        os.chdir(basedir)

