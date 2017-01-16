#!/usr/bin/env python3
#
#
"""@package plotting

A sample plotting tool for the one-dimensional DG shock tube
problems. The exact solution is from an exact Riemann problem solver

"""
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
import scipy as sp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axis as axis


codedir = '..'
sys.path.insert(0, codedir)
import solution

#================================================================================
#
# Parse arguments
#
#================================================================================
parser = argparse.ArgumentParser(description='A simple plot tool for the one-dimensional shock tube problems')
parser.add_argument('-s','--show', help='Show the plots', action='store_true')
parser.add_argument('-f','--file', dest='step', help='File to load', type=int, required=True)
parser.add_argument('-d','--directory', dest='fdir', help='Directory containing data', type=str, required=True)
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
# Problem setup
#
#================================================================================

# change directory
os.chdir(args.fdir)

# get the solution
solution = solution.Solution('empty','euler',0)
solution.loader(args.step)

# Ignore ghost cells and take only the cell average
sol = solution.u[0,solution.N_F:-solution.N_F]

# Get the primitive variables for Euler
gamma  = 1.4
rho    = sol[0::solution.N_F]
u      = sol[1::solution.N_F]/rho
p      = (gamma-1)*(sol[2::solution.N_F] - 0.5*rho*u*u)

# Exact solution
dat = np.loadtxt('exact.txt',delimiter=',')
xe   = dat[:,0]
rhoe = dat[:,1]
ue   = dat[:,2]
pe   = dat[:,3]

# Density
plt.figure(0)
plt.plot(solution.xc,rho,'o',mfc=cmap[0],mec=cmap[0])
plt.plot(xe,rhoe,color=cmap[-1],lw=2)
ax = plt.gca()
plt.xlabel(r"x",fontsize=22,fontweight='bold')
plt.ylabel(r"\rho",fontsize=22,fontweight='bold')
plt.setp(ax.get_xmajorticklabels(),fontsize=18,fontweight='bold');
plt.setp(ax.get_ymajorticklabels(),fontsize=18,fontweight='bold');
plt.savefig('density.png',format='png')

# Velocity
plt.figure(1)
plt.plot(solution.xc,u,'o',mfc=cmap[0],mec=cmap[0])
plt.plot(xe,ue,color=cmap[-1],lw=2)
ax = plt.gca()
plt.xlabel(r"x",fontsize=22,fontweight='bold')
plt.ylabel(r"u",fontsize=22,fontweight='bold')
plt.setp(ax.get_xmajorticklabels(),fontsize=18,fontweight='bold');
plt.setp(ax.get_ymajorticklabels(),fontsize=18,fontweight='bold');
plt.savefig('velocity.png',format='png')

# Pressure
plt.figure(2)
plt.plot(solution.xc,p,'o',mfc=cmap[0],mec=cmap[0])
plt.plot(xe,pe,color=cmap[-1],lw=2)
ax = plt.gca()
plt.xlabel(r"x",fontsize=22,fontweight='bold')
plt.ylabel(r"p",fontsize=22,fontweight='bold')
plt.setp(ax.get_xmajorticklabels(),fontsize=18,fontweight='bold');
plt.setp(ax.get_ymajorticklabels(),fontsize=18,fontweight='bold');
plt.savefig('pressure.png',format='png')


sname = 'sensor0000000001.dat'
if os.path.isfile(sname):
    sen = np.loadtxt(sname,delimiter=',')

    plt.figure(3)
    plt.plot(sen[:,0],sen[:,1],'o',mfc=cmap[0],mec=cmap[0])
    ax = plt.gca()
    plt.xlabel(r"x",fontsize=22,fontweight='bold')
    plt.ylabel(r"s",fontsize=22,fontweight='bold')
    plt.setp(ax.get_xmajorticklabels(),fontsize=18,fontweight='bold');
    plt.setp(ax.get_ymajorticklabels(),fontsize=18,fontweight='bold');
    plt.savefig('sensors.png',format='png')


if args.show:
    plt.show()
