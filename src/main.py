#!/usr/bin/env python
#
#
"""@package dg1d

A simple one-dimensional Discontinuous Galerkin solver.

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

import helpers
import deck
import solution
import rk
import dg

#================================================================================
#
# Parse arguments
#
#================================================================================
parser = argparse.ArgumentParser(description='A simple one-dimensional Discontinuous Galerkin solver.')
parser.add_argument('-d','--deck', help='Name of input deck file', default="deck.inp")
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

# Parse the deck
deck = deck.Deck()
deck.parser(args.deck)

# Generate the solution and apply the initial condition
sol = solution.Solution(deck.ic,deck.system,deck.order)
sol.apply_ic()

# Initialize the DG solver
dgsolver = dg.DG(sol)

#================================================================================
#
# Solve the problem
#
#================================================================================
print("Integrating the solution in time.")
rk.integrate(sol,deck,dgsolver)
