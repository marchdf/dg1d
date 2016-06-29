#!/usr/bin/env python
#
#
__author__ = 'Marc T. Henry de Frahan'
__copyright__ = "Copyright (C) 2016, Regents of the University of Michigan"
__license__ = "GPL"
__email__ = "marchdf@umich.edu"
__status__ = "Development"

#================================================================================
#
# Imports
#
#================================================================================
import argparse
import sys, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axis as axis

#================================================================================
#
# Parse arguments
#
#================================================================================
parser = argparse.ArgumentParser(description='A simple plot tool for the one-dimensional DG data')
parser.add_argument('-s','--show', help='Show the plots', action='store_true')
args = parser.parse_args()

#================================================================================
#
# Some defaults variables
#
#================================================================================
plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times')
cmap_med =['#F15A60','#7AC36A','#5A9BD4','#FAA75B','#9E67AB','#CE7058','#D77FB4','#737373']
cmap =['#EE2E2F','#008C48','#185AA9','#F47D23','#662C91','#A21D21','#B43894','#010202']
dashseq = [(None,None),[10,5],[10, 4, 3, 4],[3, 3],[10, 4, 3, 4, 3, 4],[3, 3],[3, 3]];
markertype = ['s','d','o','p','h']

#================================================================================
#
# Function definitions
#
#================================================================================
def error_norm(u0,uf,etype='L2'):
    """Returns the cell-average error norm for DG between u0 and uf"""

    if etype == 'L2':
        return np.sqrt(sum((0.5*u0e - 0.5*u0f)**2)/res)

    elif etype == 'L1':
        return sum(np.fabs(0.5*u0e - 0.5*u0f))/len(u0e)

    elif etype == 'Linf':
        return max(np.fabs(0.5*u0e - 0.5*u0f))

    else:
        print('Wrong error norm specified, defaulting to L2 norm')
        return np.sqrt(sum((0.5*u0e - 0.5*u0f)**2)/res)
        

#================================================================================
#
# Problem setup
#
#================================================================================

orders = [2,3,4]

#================================================================================
#
# Plot the cell average errors for these orders
#
#================================================================================
for p, order in enumerate(orders):

    # Get the resolutions and sort in float order
    resdirs = os.listdir(str(order))
    resolutions = [int(i) for i in resdirs]
    resolutions.sort()

    # Initialize variables
    dxs    = 2.0/np.array(resolutions)
    errors = np.zeros(len(resolutions))
    
    for k,res in enumerate(resolutions):

        ppdir = str(order)+'/'+str(res)
        print(ppdir)
        
        # Get the initial (exact solution)
        exact = np.loadtxt(ppdir+'/u0000000000.dat',delimiter=',')
        xce = exact[:,0]
        u0e = exact[:,1]
        
        # Get the final solution
        final = np.loadtxt(ppdir+'/u0000000010.dat',delimiter=',')
        xcf = final[:,0]
        u0f = final[:,1]

        # Get the cell-average error
        errors[k] = error_norm(u0e,u0f,'L2')
    
        
    # Plot the errors
    plt.figure(1)
    plt.loglog(resolutions,errors,markertype[p],color=cmap[p],mec=cmap[p],mfc=cmap[p],lw=2,ls='-',ms=10)

    # theoretical error
    th = dxs**(2*order+1)*errors[1]/dxs[1]**(2*order+1)
    plt.loglog(resolutions,th,color=cmap[-1],lw=2,ls='-')
    ax = plt.gca()
    plt.setp(ax,ylim=[1e-13,1e0])


if args.show:
    plt.show()
