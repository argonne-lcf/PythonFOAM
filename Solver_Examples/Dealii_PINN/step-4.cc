/* ---------------------------------------------------------------------
 *
 * This is a modified version of the Step-4 Deal.ii tutorial code- 
 * https://dealii.org/developer/doxygen/deal.II/step_4.html
 * 
 * This code utilizes Deal.ii to solve the Poisson equation with homogenous Dirichlet BCs
 * and transfers the data over to Python, to train a Physics Informed Neural Network (PINN)
 * using PyTorch.
 * ---------------------------------------------------------------------
*/


#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <sys/resource.h>
#include <vector>

// 
// 
// 
// 
#include <deal.II/base/logstream.h>

// 
// 
using namespace dealii;

void init_numpy() {
  import_array1();
}

// 

// 
template <int dim>
class Step4
{
public:
  Step4();
  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
  Vector<double> coordinatex;
  Vector<double> coordinatey;
};





template <int dim>
class RightHandSide : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override;
};



template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override;
};

//  Forcing function definition. Function chosen = $x^2 + y^2$
template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
                                 const unsigned int /*component*/) const
{
  double return_value = 0.0;
  for (unsigned int i = 0; i < dim; ++i)
    return_value += std::pow(p(i), 4.0);

  return return_value;
}


//  Homogenous Dirichlet BC implementation-

template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p,
                                  const unsigned int /*component*/) const
{
  return 0;
}



// Class implementation

// 
template <int dim>
Step4<dim>::Step4()
  : fe(1)
  , dof_handler(triangulation)
{}


// Grid Generation.

template <int dim>
void Step4<dim>::make_grid()
{
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(6);

  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: " << triangulation.n_cells()
            << std::endl;
}

// Generate Sparsity pattern

template <int dim>
void Step4<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
  coordinatex.reinit(dof_handler.n_dofs());
  coordinatey.reinit(dof_handler.n_dofs());
}


// Element assembly


template <int dim>
void Step4<dim>::assemble_system()
{
  QGauss<dim> quadrature_formula(fe.degree + 1);

  // 
  RightHandSide<dim> right_hand_side;

  // 
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  //
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  Point<dim> vertex_data;
  // 
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      // Global stiffness matrix assembly
      // 
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        for (const unsigned int i : fe_values.dof_indices())
          {
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx

            const auto &x_q = fe_values.quadrature_point(q_index);
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            right_hand_side.value(x_q) *        // f(x_q)
                            fe_values.JxW(q_index));            // dx
          }
      // RHS assembly
      
      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices())
        {
          for (const unsigned int j : fe_values.dof_indices())
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));

          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
        
      
      for (auto vertex_index : GeometryInfo<dim>::vertex_indices())
      {
           
           vertex_data = cell->vertex(vertex_index);
           
           int val ;
           val = local_dof_indices[vertex_index];
          
           coordinatex[val] = vertex_data[0];
            
           coordinatey[val] = vertex_data[1];



      }

    }

  // 
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           BoundaryValues<dim>(),
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);

 
}


// @sect4{Step4::solve}


template <int dim>
void Step4<dim>::solve()
{
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  // We have made one addition, though: since we suppress output from the
  // linear solvers, we have to print the number of iterations by hand.
  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;

  
}


// This section is where the code deviates from the Deal.ii tutorial example, where we send data to Python.
// We send the Finite Element solution as a n_DoFs sized vector, the DoF coordinates as a n_DoFs x 2 array and 
// additionally another array containing boundary node DoF indices to the Python Script.

// 
template <int dim>
void Step4<dim>::output_results() const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  
  PyObject *pName, *pModule, *pruntrain;
  
  double coords[dof_handler.n_dofs()][2];
  for(int l = 0; l < dof_handler.n_dofs();l++)
  {
    coords[l][0] = 0;
    coords[l][1] = 0;
  }
  
  
  for (int ind = 0; ind < dof_handler.n_dofs(); ind++)
  {
     
    
     coords[ind][0] = coordinatex[ind];
     coords[ind][1] = coordinatey[ind];
      
  }

  // Loop to transfer coordinates as per global DoF numbering-
  
  









  const IndexSet boundary_dofs = DoFTools::extract_boundary_dofs(dof_handler);
  std::vector<IndexSet::size_type> BDoFs;
  boundary_dofs.fill_index_vector(BDoFs);
  double boundarydofs[dof_handler.n_boundary_dofs()];
  for (int ind2 = 0; ind2 < dof_handler.n_boundary_dofs(); ind2++)
  {
    boundarydofs[ind2] = BDoFs[ind2];
    //cout<<boundarydofs[ind2]<<std::endl;
  }
  cout<<dof_handler.n_boundary_dofs()<<std::endl;

  Py_Initialize();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\".\")");

  
  // initialize numpy array library
  init_numpy();
  std::cout << "Initialized numpy library" << std::endl;  
  
  
  std::cout << "Loading python module" << std::endl;
  pName = PyUnicode_DecodeFSDefault("python_module"); // Python filename
  pModule = PyImport_Import(pName);
  std::cout << "Loaded python module" << std::endl;

  std::cout << "Loading functions from module" << std::endl;
  
  pruntrain = PyObject_GetAttrString(pModule, "run_training");
  std::cout << "Loaded training function from module" << std::endl;

  double solutionval[dof_handler.n_dofs()];
  for (int i=0; i <= dof_handler.n_dofs(); i++){
    
    solutionval[i] = solution[i] ;
  }
  
  PyObject *pArgs, *array_1d, *array_2d,*array_1d2;
  PyArrayObject *pValue;
  pArgs = PyTuple_New(3);
  
  //Numpy array dimensions
  npy_intp dim1[]  = {4225};
  npy_intp dim2[] = {4225,2};
  npy_intp dim3[] = {256};
  
  // create a new array
  
  array_1d = PyArray_SimpleNewFromData(1, dim1, NPY_FLOAT64, solutionval);
  
  array_2d = PyArray_SimpleNewFromData(2, dim2, NPY_FLOAT64, coords);
  
  array_1d2 = PyArray_SimpleNewFromData(1, dim3, NPY_FLOAT64, boundarydofs);
  
  PyTuple_SetItem(pArgs, 0, array_1d);
  PyTuple_SetItem(pArgs, 1, array_2d);
  PyTuple_SetItem(pArgs, 2, array_1d2);
  std::cout << "Calling python function"<<std::endl;
  pValue = (PyArrayObject*)PyObject_CallObject(pruntrain, pArgs); //Casting to PyArrayObject
  std::cout << "Called python analyses function successfully"<<std::endl;

  //Py_DECREF(pArgs);
  //PyArray_ENABLEFLAGS((PyArrayObject*)array_1d, NPY_ARRAY_OWNDATA); // Deallocate array_1d
  //PyArray_ENABLEFLAGS((PyArrayObject*)array_2d, NPY_ARRAY_OWNDATA); 
  //Py_DECREF(pValue);
  



  data_out.build_patches();

  std::ofstream output(dim == 2 ? "solution-2d.vtk" : "solution-3d.vtk");
  data_out.write_vtk(output);
}



// @sect4{Step4::run}


template <int dim>
void Step4<dim>::run()
{
  std::cout << "Solving problem in " << dim << " space dimensions."
            << std::endl;

  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
}


// This code utilizes templates and can also solve the 3D poisson equation, however you will need to make changes to the coords field such that it
// can hold the x, y and z coordinates.

int main()
{
  {
    Step4<2> laplace_problem_2d;
    laplace_problem_2d.run();
  }

  //{
  //  Step4<3> laplace_problem_3d;
  //  laplace_problem_3d.run();
  //}

  return 0;
}
