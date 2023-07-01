## PoissonFOAM

This is an implementation of a FV-PINN (see https://doi.org/10.1063/5.0097480) implemented by Deepinder Jot Singh Aulakh - ALCF Research Aide at Argonne National Laboratory.

A tensorflow-based optimizer identifies the solution X to AX = B within Python after the system matrix for the Poisson equation (A) is provided to it by OpenFOAM. This is a prototypical example of how a linear system solve can be performed on the Python side of PythonFOAM. 