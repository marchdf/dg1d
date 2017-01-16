# Regression test suite

Execute `nosetests regressions/regressions.py` from the code directory
to run the regression test suite.

## Test cases

The domain is [-1, 1], the ratio of specific heats is 1.4, the CFL is
0.5, and the number of elements is 100. The exact solution was
generated with an exact Riemann solver.

Test | x0 | End time | rhoL | uL | pL | rhoR | uR | pR
-----|----|----------|------|----|----|------|----|---
Sod shock tube                         |  0   | 0.2   |       1|         0|       1|   0.125|     0.0|     0.1
Modified Sod shock tube                | -0.4 | 0.2   |       1|      0.75|     1.0|   0.125|     0.0|     0.1
123 problem                            |  0   | 0.15  |       1|      -2.0|     0.4|     1.0|     2.0|     0.4
Blast wave (Left Woodward and Colella) |  0   | 0.012 |       1|       0.0|  1000.0|     1.0|     0.0|    0.01
collision of two strong shocks         | -0.2 | 0.035 | 5.99924|   19.5975| 460.894| 5.99242|-6.19633| 46.0950
stationary contact discontinuity       |  0   | 2.0   |     1.4|       0.0|     1.0|     1.0|     0.0|     1.0


## Sod Shock tube
Code options can be found [here](./sodtube/deck.inp).

<img src="./sodtube/density.png" alt="density.png" width="400">
<img src="./sodtube/velocity.png" alt="velocity.png" width="400">
<img src="./sodtube/pressure.png" alt="pressure.png" width="400">

# Modified Sod shock tube
Code options can be found [here](sodtube_modified/deck.inp).

<img src="./sodtube_modified/density.png" alt="density.png" width="400">
<img src="./sodtube_modified/velocity.png" alt="velocity.png" width="400">
<img src="./sodtube/pressure.png" alt="pressure.png" width="400">

# 123 problem
Code options can be found [here](123_problem/deck.inp).

<img src="./123_problem/density.png" alt="density.png" width="400">
<img src="./123_problem/velocity.png" alt="velocity.png" width="400">
<img src="./123_problem/pressure.png" alt="pressure.png" width="400">

# Blast wave (Left Woodward and Colella)
Code options can be found [here](blast_wave/deck.inp).

<img src="./blast_wave/density.png" alt="density.png" width="400">
<img src="./blast_wave/velocity.png" alt="velocity.png" width="400">
<img src="./blast_wave/pressure.png" alt="pressure.png" width="400">

# Collision of two strong shocks
Code options can be found [here](strong_shocks/deck.inp).

<img src="./strong_shocks/density.png" alt="density.png" width="400">
<img src="./strong_shocks/velocity.png" alt="velocity.png" width="400">
<img src="./strong_shocks/pressure.png" alt="pressure.png" width="400">

# Stationary contact discontinuity 
Code options can be found [here](stationary_contact/deck.inp).

<img src="./stationary_contact/density.png" alt="density.png" width="400">
<img src="./stationary_contact/velocity.png" alt="velocity.png" width="400">
<img src="./stationary_contact/pressure.png" alt="pressure.png" width="400">
