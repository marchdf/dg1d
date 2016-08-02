#================================================================================
#
# Imports
#
#================================================================================
import sys
import numpy as np

import solution

#================================================================================
#
# Class definitions
#
#================================================================================
class Limiter:
    'Generate the limiter'

    #================================================================================
    def __init__(self,limiting_type,solution):

        print("Setting up the limiter.")

        # Pre-allocate vector holding 1/(2*k+1). The little None
        # business is to do the multiplication broadcasting with the
        # delta. According to
        # http://stackoverflow.com/questions/16229823/how-to-multiply-numpy-2d-array-with-numpy-1d-array
        self.c = 1.0/(2*np.arange(0,solution.basis.p) + 1)[:, None]

        # By default, do not limit
        self.do_limiting = False 
        
        # Pre-allocate some vectors when doing limiting everywhere in
        # the domain
        if limiting_type == 'full':
            print('Limiting everywhere')
            self.do_limiting = True
            
        elif limiting_type == 'adaptive':
            print('Adaptive limiting')
            self.do_limiting = True
            
        else:
            print('No limiting.')
            
            
    #================================================================================
    def limit(self,solution):
        """Limit a solution"""

        if self.do_limiting:

            # Get the differences with the left/right neighbors
            deltam,deltap = self.deltas(solution)
        
            # Apply the minmod procedure
            solution.u[1:,solution.N_F:-solution.N_F] = self.minmod(deltam,deltap,solution.u[1:,solution.N_F:-solution.N_F])


    #================================================================================
    def deltas(self,solution):
        """Calculate the difference between left and right neightbors"""

        # Total number of elements in the solution (including ghosts)
        total_num_element = solution.u.shape[1]

        #e = np.array([1,3,7,10])
        #idx = ((e*N_F).reshape((len(e),1)) +np.arange(solution.N_F)).flatten() 

        # Index of all elements to be limited
        idx = np.arange(solution.N_F,total_num_element-solution.N_F)

        # Index of their neighbors to the left
        idxl = idx - solution.N_F 

        # Index of their neighbors to the right
        idxr = idx + solution.N_F

        # Differences with the left and right neighbors
        deltam = (solution.u[:-1,idx]  - solution.u[:-1,idxl]) *self.c
        deltap = (solution.u[:-1,idxr] - solution.u[:-1,idx])  *self.c

        return deltam,deltap
        

    #================================================================================
    def minmod(self,A,B,C):
        """Given three arrays do an element-by-element minmod

        For each entry a, b, c in A, B,C, return:
           max(a,b,c) if a,b,c < 0
           min(a,b,c) if a,b,c > 0
           0          otherwise

        There is another way of doing this. Here, we are getting the
        max and min everywhere in the array, then deciding where to
        used the min/max (depending on the sign of the
        elements). Another way would be to find the indices where the
        min/max should be taken, then take the min/max for those
        columns. I ran some tests and (surprisingly?) the faster way
        is actually to do it the first way (min/max then decide where
        to use them). I left the other way below commented.
        """

       
        # Initialize
        M = np.zeros(A.size)
        D = np.vstack((A.flatten(),B.flatten(),C.flatten()))

        # Find where indices where they are all positive or all
        # negative and we should be getting the max or min
        idxmax = np.where(np.all(D<0,axis=0))
        idxmin = np.where(np.all(D>0,axis=0))

        # Find the max and min comparing a,b,c
        maxi = np.max(D,axis=0)
        mini = np.min(D,axis=0)
        
        # Populate the minmod matrix with the results
        M[idxmax] = maxi[idxmax]
        M[idxmin] = mini[idxmin]

        # I initially thought this might be faster. But it's not. I am
        # leaving this here because this result is a bit surprising to
        # me. In theory, you are doing fewer min/max operations
        # compared to the other way.         
        # # Find the max and min comparing a,b,c
        # maxi = np.max(D[:,idxmax],axis=0)
        # mini = np.min(D[:,idxmin],axis=0)
        # # Populate the minmod matrix with the results
        # M[idxmax] = maxi
        # M[idxmin] = mini

        # Reshape the matrix and return the results
        return M.reshape(A.shape)
