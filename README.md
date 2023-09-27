# An algorithm to solve quasi-linear screening problems

This repository provides code to solve quasi-linear screening problems by a Primal-Dual Hybrid Gradient algorithm (PDHG, a.k.a Chambolle-Pock) and to reproduce the numerical illustrations of the paper *A General Solution to the Quasi Linear Screening Problem* by G. Carlier, X. Dupuis, J.-C. Rochet, and J. Thanassoulis.

The `modules` folder contains the code itself with two models:

- `PAP` for 2D monopolist problems;
- `TAX` for 2D taxation problems.
  
The `examples` folder contains Python scripts and Jupyter notebooks which shows how to set and solve both models.

The `environment.yml` file provides a minimal conda environment required to run the code.
