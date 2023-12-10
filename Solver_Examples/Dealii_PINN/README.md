This solver example demonstrates a Deal.ii-PINN coupling where a Deal.ii FEM code solves the Poisson equation over a unit square, with forcing function $x^2+y^2$, with
homogenous Dirichlet BCs on all edges. The solution, boundary DoF information and nodal coordinates are then sent over to Python_Module, where the data is used 
to train a PINN using PyTorch. The C++ FEM code is based on the Deal.ii Step-4 tutorial (https://dealii.org/developer/doxygen/deal.II/step_4.html). 
In the Deal.ii code, the following changes have been made- \
Lines 238-248 in assemble_system() : Node coordinates passed on to two Dealii::Vector objects coordinatex and coordinatey, the ordering is the based on the global DoF indices. \
Lines 305-320 in output_results() : Node coordinates passed to C++ array, to transfer to Python.\
Lines 324-332 in output_results() : Boundary DoF indices passed to C++ array, to transfer to Python.\
Lines 335-353 in output_results() : Initialized Python and NumPy from within the code, loaded the Python_Module and functions inside it.\
Lines 355-359 in output_results() : Passed solution to C++ array, to transfer to Python.\
Lines 361-388 in output_results() : Passed the Boundary DoF data, Node coordinates and solution vector to the run_training function within Python_Module.\
Expected plot from running the code - \
![PINN_Output](https://github.com/SumedhSCU/PythonFOAM/assets/143654947/2ea74a14-bff0-4f8c-bb39-b2f32e10a7ad) \
The plots look wonky due to there not being enough gridpoints for matplotlib to generate a smooth contourplot, increasing the grid refinement and accordingly changing the 
network parameters in Python_Module should improve it

