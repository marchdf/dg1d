#!/usr/bin/env python3
#
#
"""@package sample plotting

A sample plotting tool for the one-dimensional DG data. The exact
solution plotted is for the sinewave initial condition.

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

import solution
import dg

#================================================================================
#
# Parse arguments
#
#================================================================================
parser = argparse.ArgumentParser(description='A simple plot tool for the one-dimensional DG data')
parser.add_argument('-s','--show', help='Show the plots', action='store_true')
parser.add_argument('-f','--file', dest='step', help='File to load', type=int, required=True)
parser.add_argument('-t','--type', dest='system', help='Type of system to solve', type=str, default='advection')
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

solution = solution.Solution('empty',args.system,0)
solution.loader(args.step)

# Collocate the solution to the Gaussian nodes
ug = solution.collocate()

# Collocate to the cell edge values
uf = solution.evaluate_faces()

# Get the primitive variables for Euler
if args.system == 'euler':
    gamma  = 1.4
    rho  = ug[:,0::solution.N_F]
    u    = ug[:,1::solution.N_F]/rho
    p    = (gamma-1)*(ug[:,2::solution.N_F] - 0.5*rho*u*u)
    ug[:,1::solution.N_F]   = u
    ug[:,2::solution.N_F]   = p

    rho  = uf[:,0::solution.N_F]
    u    = uf[:,1::solution.N_F]/rho
    p    = (gamma-1)*(uf[:,2::solution.N_F] - 0.5*rho*u*u)
    uf[:,1::solution.N_F]   = u
    uf[:,2::solution.N_F]   = p


# Plot each field
for field in range(solution.N_F):

    plt.figure(field)
    
    # Plot each element solution in a different color
    # Skip plotting the ghost cells
    for e in range(1,solution.N_E+1):
        a = solution.x[e-1]
        b = solution.x[e]
        xg = 0.5*(b-a)*solution.basis.x + 0.5*(b+a)
        
        # plot the solution at the Gaussian nodes (circles)
        plt.plot(xg,ug[:,e*solution.N_F+field],'o',mfc=cmap[e%len(cmap)],mec=cmap[e%len(cmap)])
        
        # Plot the solution at the cell edges (squares)
        plt.plot([a,b],uf[:,e*solution.N_F+field],'s',mfc=cmap[e%len(cmap)],mec='black')

    # Plot the exact solution
    if args.system == 'advection':
        xe = np.linspace(-1,1,200)
        fe = np.sin(2*np.pi*xe)
        plt.plot(xe,fe,'k')


# Plot the sensors if they exist
if os.path.isfile('sensor0000000000.dat'):
    plt.figure(solution.N_F)
    dat = np.loadtxt('sensor{0:010d}.dat'.format(args.step),delimiter=',')

    # plot sensors at cell centers
    plt.plot(dat[:,0],dat[:,1],'o',mfc=cmap[0],mec=cmap[0])
    plt.ylim([-0.1,2.1])
        
if args.show:
    plt.show()
