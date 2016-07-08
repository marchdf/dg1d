#================================================================================
#
# Imports
#
#================================================================================
import sys
import re
import numpy as np
import basis
import copy

from numpy.polynomial import Polynomial as P
from numpy.polynomial import legendre as leg # import the Legendre functions
from numpy.polynomial import Legendre as L   # import the Legendre class


#================================================================================
#
# Class definitions
#
#================================================================================
class Enhance:
    'Generate enhancement procedures'

    #================================================================================
    def __init__(self,solution_order,method):

        print("Generating the enhancement procedure.")
        
        # Parse the enhancement type
        method = method.split()
        self.etype = method[0]
        self.modes = [int(i) for i in method[1:] ];
        # sort modes (makes the indexing cleaner in later functions)
        self.modes.sort() 
        
        # enhancement order
        self.order = solution_order + len(self.modes)
        
        # The enhanced basis
        self.basis = basis.Basis(self.order)
        #self.basis = basis.Basis(solution_order)

        
        A,Ainv,B,Binv = enhancement_matrices(solution_order,self.modes)
        self.alphaL, self.alphaR, self.betaL, self.betaR = left_enhancement_vectors(Ainv,Binv,solution_order,self.modes,self.basis.psi)
        print('bye')



    #================================================================================
    def face_value(self,u):
        """Calculates the value of the enhanced solution at the faces"""

        print(u)
        

        return 0
        #return np.dot(self.basis.psi, u)
            


#================================================================================
def left_enhancement_vectors(Ainv,Binv,solution_order,modes,psi):
    """Returns the enhancement vectors
    
    These are the vectors that we apply to the solution (in the left
    and right cell) to get the value of the solution in the left cell
    at the interface between the left and right cell.

    See notes 2016/07/06 page 4 for more details.
    """

    # Multiply by the basis evaluated at the start and end points [-1,1]
    vs = np.dot(psi[0,:],Binv)
    ve = np.dot(psi[1,:],Ainv)

    print(Binv)
    
    # Split to get the vector acting on the left solution
    betaR  = vs[:solution_order+1]
    alphaL = ve[:solution_order+1]

    # Split to get the vector acting on the right solution
    betaL  = vs[solution_order+1:]
    alphaR = ve[solution_order+1:]
    # And add zeros for the modes of the right solution that we want
    # to leave out of the enhancement procedure
    for i in range(solution_order+1):
        if i not in modes:
            betaL  = np.insert(betaL ,i,0)
            alphaR = np.insert(alphaR,i,0)
    
    return alphaL,alphaR,betaL,betaR

#================================================================================
def enhancement_matrices(solution_order,modes):
    """Returns the enhancement matrices (and their inverse)
    
    Returns A and inv(A) where A \hat{u} = [uL;some_modes_of(uR)]
            B and inv(B) where B \hat{u} = [uR;some_modes_of(uL)]
    """

    # Enhanced solution order
    order = solution_order + len(modes)
    
    # Submatrices to build the main matrix later
    a  = np.diag(np.ones(solution_order+1))
    b  = np.zeros((solution_order+1,len(modes)))
    cl = np.zeros((len(modes),order+1))
    cr = np.zeros((len(modes),order+1))
    
    # Loop on the modes we are keeping in the neighboring cell
    # (the right cell)
    for i,mode in enumerate(modes):
        
        # Loop on the enhancement basis 
        for j in range(order+1):

            # Basis function in the right cell
            l1 = L.basis(mode)
            
            # Enhanced basis function extending into the right cell (or left cell)
            ll = basis.shift_legendre_polynomial(L.basis(j),2)
            lr = basis.shift_legendre_polynomial(L.basis(j),-2)
            
            # Inner product for the left and right enhancements
            cl[i,j] = basis.integrate_legendre_product(l1,ll) / basis.integrate_legendre_product(l1,l1)
            cr[i,j] = basis.integrate_legendre_product(l1,lr) / basis.integrate_legendre_product(l1,l1)
                
    # Put the matrices together
    A = np.vstack((np.hstack((a,b)),cl))
    B = np.vstack((np.hstack((a,b)),cr))
    return A, np.linalg.inv(A), B, np.linalg.inv(B)

