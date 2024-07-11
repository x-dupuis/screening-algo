# An algorithm to solve quasi-linear screening problems

This repository provides code to solve quasi-linear screening problems by a Primal-Dual Hybrid Gradient algorithm (PDHG, a.k.a Chambolle-Pock) and to reproduce the numerical illustrations of the paper *A General Solution to the Quasi Linear Screening Problem* by G. Carlier, X. Dupuis, J.-C. Rochet, and J. Thanassoulis (Journal of Mathematical Economics, 2024).

The `modules` folder contains the code itself with:

- a base class `Screening`
  
and three models given as derived classes:

- `PAP` for 2D monopolist problems;
- `LinearMonopolist` for 2D linear monopolist problems (or Schmalensee problems);
- `TAX` for 2D taxation problems.

The `examples` folder contains Python scripts and Jupyter notebooks which show how to set and solve the different models.

The `environment.yml` file provides a minimal conda environment required to run the code.
