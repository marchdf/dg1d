# One-dimensional Discontinuous Galerkin code

This is a python implementation of the one-dimensional Discontinuous
Galerkin method to solve 

a) a simple linear advection partial differential equation;
b) the Euler equations. 

There are a lot of assumptions that go into making this code as simple
as possible, including the assumption that the grid is
one-dimensional, structured, and the elements are of constant size.
