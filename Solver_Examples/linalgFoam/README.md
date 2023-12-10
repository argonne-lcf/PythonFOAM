This solver is a variation of LaplacianFoam, where the stiffness matrix created within OpenFOAM and the forcing vector, as well as the solution are passed to a C++ array, which is then
sent to Python as numpy arrays. This arrangement allows one to compare multiple solvers (NumPy, PETSc etc.). Since an explicit full matrix isn't stored in OpenFOAM for 
efficiency reasons, the LDU (Lower-Diagonal-Upper) addressing system used by OpenFOAM was utilized to create the matrices and vectors. The changes to the source code of laplacianFoam are thus-
laplacianFoam are as follows- \
Lines 65-84 : Invoking Python and NumPy, and loading Python_Module and required functions from within it. \
Lines 86-91 : Creating tuple to pass function arguments.\
Lines 124-131: Getting face addresses and lower diagonal, upper diagonal and diagonal components as scalarField objects.\
Lines 137-161: Passing lower diagonal, upper diagonal and diagonal components to simpleMatrix object based on face addressing.\
Lines 163-176: Adding boundary contributions.\
Lines 183-192: Passing the created stiffness matrices and forcing vector to C++ array objects, to be converted to NumPy arrays and sent to Python.\
Lines 199-202: Passing the solution vector to C++ array object, to be converted to NumPy array and sent to Python.

The residual plot obtained from Python should look similar to this:\
![Residual_Plot](https://github.com/SumedhSCU/PythonFOAM/assets/143654947/f0e4269e-4aa8-4614-a900-858695a6262c)



