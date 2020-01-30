/********************************************************************************
Stokes 2D Example
********************************************************************************/
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/fe/component_mask.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/identity_matrix.h>
#include <deal.II/lac/block_matrix_array.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/base/parameter_handler.h>

#include <cmath>
#include <fstream>
#include <iostream>

#include "PDASProblem.hpp"
#include "PDASStokes.hpp"
#include "PDASLaplace.hpp"

using namespace dealii;

int main(int argc, char* argv[]){

    Timer timer;
    timer.start();

    const std::string params_file=argv[2];
    const int dim=2;

    PDASStokes<dim> problem(params_file);
    problem.run();

  timer.stop();
  std::cout << "TOT Elapsed CPU time: " << timer.cpu_time() << " seconds." << std::endl;


}
