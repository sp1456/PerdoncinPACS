/********************************************************************************
PDASLaplace Class for solving Laplace OCPs
********************************************************************************/

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/component_mask.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>

#include <cmath>
#include <fstream>
#include <iostream>


using namespace dealii;

template <int dim>
class PDASLaplace: public PDASProblem<dim> {
public:
   PDASLaplace(std::string file): PDASProblem<dim>(file,1) {};

private:
  virtual void dof_renumbering() override;
  virtual void assemble_system() override;
  virtual void output_result() override;

};


template<int dim>
void PDASLaplace<dim>::dof_renumbering(){
  //Renumber dofs with Reverse Cuthill McKee algorithm
  DoFRenumbering::Cuthill_McKee(this->dof_handler,true);
}


template <int dim>
void PDASLaplace<dim>::assemble_system(){
//Assemble all blocks of the system that do not depend on the iteration of the optimization cycle

//Assembly of vectors z,umax and umin by interpolating in the nodes the corresponding functions
  Vector<double> z(this->dof_handler.n_dofs());
  VectorTools::interpolate(this->dof_handler,ScalarFunction<dim>(this->z_expr), z);
  VectorTools::interpolate(this->dof_handler,ScalarFunction<dim>(this->b_expr), this->umax);
  VectorTools::interpolate(this->dof_handler,ScalarFunction<dim>(this->a_expr), this->umin);



//LHS and RHS assembly

  MatrixCreator::create_laplace_matrix(this->dof_handler,
                                        QGauss<dim>(this->fe->degree + 1),
                                        this->LHS.block(0,2));
  MatrixCreator::create_mass_matrix(this->dof_handler,
                                    QGauss<dim>(this->fe->degree + 1),
                                    this->LHS.block(0,0));
  VectorTools::create_right_hand_side(this->dof_handler,
                                      QGauss<dim>(this->fe->degree + 1),
                                      ScalarFunction<dim>(this->f_expr),
                                      this->RHS.block(2));

  this->LHS.block(2,0).copy_from(this->LHS.block(0,2));

  //Apply BC to state/adjoint matrices:
    std::map<types::global_dof_index, double> boundary_values;
    std::map<types::global_dof_index, double> boundary_values_dual;
    VectorTools::interpolate_boundary_values(this->dof_handler,0,
                                      ScalarFunction<dim>(this->bc_expr),
                                      boundary_values);
    VectorTools::interpolate_boundary_values(this->dof_handler,0,
                                      ZeroFunction<dim>(),
                                      boundary_values_dual);

    MatrixTools::apply_boundary_values(boundary_values,
                                    this->LHS.block(2,0),
                                    this->solution.block(2),
                                    this->RHS.block(2));
    MatrixTools::apply_boundary_values(boundary_values_dual,
                                    this->LHS.block(0,2),
                                    this->solution.block(0),
                                    this->RHS.block(0));


  //Apply BC to mass matrix
     std::vector<bool> boundary_dofs_vec(this->dof_handler.n_dofs(), false);
     DoFTools::extract_boundary_dofs(this->dof_handler,ComponentMask(),boundary_dofs_vec); //extract boundary dofs
     for (unsigned int i=0; i<this->dof_handler.n_dofs(); i++)
      if (boundary_dofs_vec[i])                                    //if it is a boundary dofs
        for (unsigned int j=0; j<this->dof_handler.n_dofs(); j++) //force the full row to zero
           if (this->LHS.block(0,0).el(i,j)!=0)
               this->LHS.block(0,0).set(i,j,0);


 this->LHS.block(2,1).copy_from(this->LHS.block(0,0));
 this->LHS.block(2,1)*=-1;

 for(unsigned int i=0; i<this->dof_handler.n_dofs(); i++){  //second row assembly
   this->LHS.block(1,1).set(i,i,this->alpha);
   this->LHS.block(1,2).set(i,i,-1);
   this->LHS.block(1,3).set(i,i,1);
  }

 this->LHS.block(0,0).vmult(this->RHS.block(0),z);

}



template <int dim>
void PDASLaplace<dim>::output_result() {
//create .vtk output files and compute norms/errors

//.vtk file for the control
  DataOut<dim> data_out_u;
  data_out_u.attach_dof_handler(this->dof_handler);
  data_out_u.add_data_vector(this->solution.block(1), "control");
  data_out_u.build_patches();
  std::ofstream output_u("solution_control.vtk");
  data_out_u.write_vtk(output_u);

//.vtk file for the state
  DataOut<dim> data_out_y;
  data_out_y.attach_dof_handler(this->dof_handler);
  data_out_y.add_data_vector(this->solution.block(0), "state");
  data_out_y.build_patches();
  std::ofstream output_y("solution_state.vtk");
  data_out_y.write_vtk(output_y);

//.vtk file for the adjoint
  DataOut<dim> data_out_p;
  data_out_p.attach_dof_handler(this->dof_handler);
  data_out_p.add_data_vector(this->solution.block(2), "adjoint");
  data_out_p.build_patches();
  std::ofstream output_p("solution_adjoint.vtk");
  data_out_p.write_vtk(output_p);

//Errors/norms computation

  //state error:
  Vector<float> difference_per_cell(this->triangulation.n_active_cells());
  VectorTools::integrate_difference(this->dof_handler,
                                    this->solution.block(0),
                                    ScalarFunction<dim>(this->z_expr),
                                    difference_per_cell,
                                    QGauss<dim>(this->fe->degree + 1),
                                    VectorTools::L2_norm);
  const double state_error =
    VectorTools::compute_global_error(this->triangulation,
                                      difference_per_cell,
                                      VectorTools::L2_norm);
  //control norm
  Vector<float> difference_per_cell2(this->triangulation.n_active_cells());
  VectorTools::integrate_difference(this->dof_handler,
                                    this->solution.block(1),
                                    ZeroFunction<dim>(),
                                    difference_per_cell2,
                                    QGauss<dim>(this->fe->degree + 1),
                                    VectorTools::L2_norm);
  const double u_norm =VectorTools::compute_global_error(this->triangulation,
                                    difference_per_cell2,
                                    VectorTools::L2_norm);
  //adjoint norm
  Vector<float> difference_per_cell3(this->triangulation.n_active_cells());
  VectorTools::integrate_difference(this->dof_handler,
                                    this->solution.block(2),
                                    ZeroFunction<dim>(),
                                    difference_per_cell3,
                                    QGauss<dim>(this->fe->degree + 1),
                                    VectorTools::L2_norm);
  const double p_norm =VectorTools::compute_global_error(this->triangulation,
                                    difference_per_cell3,
                                    VectorTools::L2_norm);

  std::cout << "|| v-z || = " << state_error << std::endl;
  std::cout << " || u ||  = " << u_norm << std::endl;
  std::cout << " || p ||  = " << p_norm << std::endl;
  std::cout << "    J     = " << 0.5*state_error*state_error
                                  + 0.5*this->alpha*u_norm*u_norm << std::endl;


}
