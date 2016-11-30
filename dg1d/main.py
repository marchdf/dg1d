#!/usr/bin/env python3
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
import numpy as np
import subprocess
import time
from datetime import timedelta

import helpers
import deck
import solution
import rk
import dg
import limiting

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


#================================================================================
#
# Function definitions
#
#================================================================================
def get_git_revision_hash():
    """Returns the git version of this project"""
    return subprocess.check_output(['git', 'describe','--always'],universal_newlines=True)

#================================================================================
#
# Problem setup
#
#================================================================================
start = time.time()
print('Code version: ', get_git_revision_hash())

# Parse the deck
deck = deck.Deck()
deck.parser(args.deck)

# Generate the solution and apply the boundary conditions
sol = solution.Solution(deck.ic,deck.system,deck.order,deck.riemann,deck.enhance,deck.sensor_thresholds)
sol.apply_bc()

# Initialize the DG solver
dgsolver = dg.DG(sol)

# Initialize the limiter and limit solution if necessary
limiter = limiting.Limiter(deck.limiting,sol)
limiter.limit(sol)

#================================================================================
#
# Solve the problem
#
#================================================================================
print("Integrating the solution in time.")
rk.integrate(sol,deck,dgsolver,limiter)

# output timer
end = time.time() - start
print("Elapsed time "+str(timedelta(seconds=end)) + " (or {0:f} seconds)".format(end))
