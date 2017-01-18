# One-dimensional Discontinuous Galerkin code

This is a python implementation of the one-dimensional Discontinuous
Galerkin method to solve 

a) a simple linear advection partial differential equation;

b) the 1D Euler equations. 

There are a lot of assumptions that go into making this code as simple
as possible, including the assumption that the grid is
one-dimensional, structured, and the elements are of constant size.

# Development process

1. Create a branch for the new feature (locally):
	```{bash}
	git checkout -b feature-branch
	```

2. Develop the feature, merging changes often from the develop branch into your feature branch:
	```{bash}
	git commit -m "Developed feature"
	git checkout develop
	git pull                     # fix any identified conflicts between local and remote branches of "develop"
	git checkout feature-branch
	git merge develop            # fix any identified conflicts between "develop" and "feature-branch"
	```

3. Push feature branch to dg1d repository:
	```{bash}
	git push -u origin feature-branch
	```

4. Create a pull request on GitHub. Make sure you ask to merge with develop

# Licensing

See [LICENSE.md](LICENSE.md)

# Author contact

Marc T. Henry de Frahan (marchdf@umich.edu)

# Jenkins configuration

See [instructions](jenkins_configuration.md)
