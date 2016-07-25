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
def get_timer(fname):
    """Get the program elapsed time from the logfile"""

    with open(fname) as f:
        for line in f:
            if 'seconds' in line:
                return line.split()[4]

    
    
#================================================================================
#
# Problem setup
#
#================================================================================

orders = [1,2,3,4]

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
    timers = np.zeros(len(resolutions))
    
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

        # Get the elapsed time
        timers[k] = get_timer(ppdir+'/logfile')
        
    # Plot the errors
    print( - np.diff(np.log(errors)) / np.diff(np.log(resolutions)), 2*order + 1)
    plt.figure(1)
    plt.loglog(resolutions,errors,markertype[p],color=cmap[p],mec=cmap[p],mfc=cmap[p],lw=2,ls='-',ms=10)

    # theoretical error
    th = dxs**(2*order+1)*errors[0]/dxs[0]**(2*order+1)
    plt.loglog(resolutions,th,color=cmap[-1],lw=2,ls='-')

    # Plot the efficiency
    plt.figure(2)
    plt.loglog(timers,errors,markertype[p],color=cmap[p],mec=cmap[p],mfc=cmap[p],lw=2,ls='-',ms=10)


# Format the plot and save
plt.figure(1)
ax = plt.gca()
plt.xlabel(r"number of elements",fontsize=22,fontweight='bold')
plt.text(0, 1.05,r"$L_2$ error", transform=ax.transAxes, fontsize=22, fontweight='bold', ha='center')
plt.setp(ax.get_xmajorticklabels(),fontsize=18,fontweight='bold');
plt.setp(ax.get_ymajorticklabels(),fontsize=18,fontweight='bold');
plt.setp(ax,xlim=[2,1e4],ylim=[1e-14,1e0])
plt.yticks([1e-14,1e-11,1e-8,1e-5,1e-2])
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.savefig('convergence.pdf',format='pdf')

# Format the plot and save
plt.figure(2)
ax = plt.gca()
plt.xlabel(r"time",fontsize=22,fontweight='bold')
plt.text(0, 1.05,r"$L_2$ error", transform=ax.transAxes, fontsize=22, fontweight='bold', ha='center')
plt.setp(ax.get_xmajorticklabels(),fontsize=18,fontweight='bold');
plt.setp(ax.get_ymajorticklabels(),fontsize=18,fontweight='bold');
plt.setp(ax,xlim=[1e-1,2e2],ylim=[1e-14,1e0])
plt.yticks([1e-14,1e-11,1e-8,1e-5,1e-2])
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.savefig('efficiency.pdf',format='pdf')

if args.show:
    plt.show()
