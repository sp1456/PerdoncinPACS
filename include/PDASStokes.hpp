/********************************************************************************
PDASStokes Class for solving Stokes OCPs
********************************************************************************/

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/tria.h>

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
class PDASStokes: public PDASProblem<dim> {
public:
  PDASStokes(std::string file): PDASProblem<dim>(file,2) { };

private:
  virtual void dof_renumbering() override;
  virtual void assemble_system() override;
  virtual void output_result() override;

  int nu;            //number of dofs realted to velocity
  int np;            //number of dofs related to pressure

};


template<int dim>
void PDASStokes<dim>::dof_renumbering(){

  //dofs renumbering
  DoFRenumbering::Cuthill_McKee(this->dof_handler,true);
  std::vector<unsigned int> block_component(dim+1,0);
  block_component[dim]=1;
  DoFRenumbering::component_wise(this->dof_handler, block_component);  //group velocity nodes separately from pressure ones

  //nu and np determination
  std::vector<types::global_dof_index> dofs_per_block(dim+1);
  DoFTools::count_dofs_per_component(this->dof_handler, dofs_per_block);
  nu=dofs_per_block[0]*dim;
  np=dofs_per_block[dim];
  std::cout << nu << " (velocity dofs) + " << np << " (pressure dofs)"  << std::endl;

}



template <int dim>
void PDASStokes<dim>::assemble_system(){
//Assemble all blocks of the system that do not depend on the iteration of the optimization cycle

//Assembly of vectors z,umax and umin by interpolating in the nodes the corresponding functions
  Vector<double> z(this->dof_handler.n_dofs());
  VectorTools::interpolate(this->dof_handler,VectorialFunction<dim>(this->z_expr), z);
  VectorTools::interpolate(this->dof_handler,VectorialFunction<dim>(this->b_expr), this->umax);
  VectorTools::interpolate(this->dof_handler,VectorialFunction<dim>(this->a_expr), this->umin);


//LHS and RHS assembly "by hand"

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  QGauss<dim> quadrature_formula(this->fe->degree +1);
  FEValues<dim> fe_values(*(this->fe),
                        quadrature_formula,
                        update_values | update_quadrature_points |
                          update_JxW_values | update_gradients);
  const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();
  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_mass(dofs_per_cell, dofs_per_cell);
  Vector<double>     local_RHS(dofs_per_cell);
  Vector<double>     local_adjoint_RHS(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  VectorialFunction<dim>    right_hand_side(this->f_expr);
  const ZeroFunction<dim>           right_hand_side_adjoint;
  std::vector<Vector<double>> RHS_values(n_q_points, Vector<double>(dim + 1));
  std::vector<Vector<double>> RHS_values_adjoint(n_q_points, Vector<double>(dim + 1));
  //basis functions
  std::vector<SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
  std::vector<Tensor<1, dim>>          phi_u(dofs_per_cell);
  std::vector<double>                  div_phi_u(dofs_per_cell);
  std::vector<double>                  phi_p(dofs_per_cell);

for (const auto &cell : this->dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    local_matrix                = 0;
    local_mass                  = 0;
    local_RHS                   = 0;
    local_adjoint_RHS              = 0;
    right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                      RHS_values);
    right_hand_side_adjoint.vector_value_list(fe_values.get_quadrature_points(),
                                            RHS_values_adjoint);

    for (unsigned int q = 0; q < n_q_points; ++q)  //quadrature points: x_q
      {
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            symgrad_phi_u[k] =fe_values[velocities].symmetric_gradient(k, q);
            phi_u[k]     = fe_values[velocities].value(k,q);
            div_phi_u[k] = fe_values[velocities].divergence(k, q);
            phi_p[k]     = fe_values[pressure].value(k, q);
          }
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j <= i; ++j)
              {
                local_matrix(i, j) +=
                  (2 * (symgrad_phi_u[i] * symgrad_phi_u[j])
                   - div_phi_u[i] * phi_p[j]
                   - phi_p[i] * div_phi_u[j])
                  * fe_values.JxW(q);                         // * dx
                local_mass(i, j) +=
                  (phi_u[i] * phi_u[j])
                  * fe_values.JxW(q);                         // * dx
              }
            const unsigned int component_i =
              this->fe->system_to_component_index(i).first;
            local_RHS(i) += (fe_values.shape_value(i, q)
                             * RHS_values[q](component_i))       // * rhs(x_q))
                            * fe_values.JxW(q);                  // * dx
            local_adjoint_RHS(i)+= (fe_values.shape_value(i, q)
                             * RHS_values_adjoint[q](component_i))  // * rhs_adjoint(x_q))
                            * fe_values.JxW(q);                  // * dx
          }
      }
    //matrix filling by symmetry
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
        {
          local_matrix(i, j) = local_matrix(j, i);
          local_mass(i, j) =local_mass(j, i);
        }

    //From local to global
    cell->get_dof_indices(local_dof_indices);

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = 0; j < dofs_per_cell; ++j){
        this->LHS.block(0,0).add(local_dof_indices[i],local_dof_indices[j],
                                  local_mass(i, j));
        this->LHS.block(2,0).add(local_dof_indices[i],local_dof_indices[j],
                                   local_matrix(i, j));
       this->LHS.block(0,2).add(local_dof_indices[i],local_dof_indices[j],
                                  local_matrix(i, j));
    }

    for (unsigned int i = 0; i < dofs_per_cell; ++i){
        this->RHS.block(2)(local_dof_indices[i]) += local_RHS(i);
        this->RHS.block(0)(local_dof_indices[i]) += local_adjoint_RHS(i);
    }

}

//Apply BC:
 std::map<types::global_dof_index, double> boundary_values;
 std::map<types::global_dof_index, double> boundary_values_adjoint;
 VectorTools::interpolate_boundary_values(this->dof_handler,0,
                                  VectorialFunction<dim>(this->bc_expr),
                                  boundary_values,
                                 this->fe->component_mask(velocities));
 VectorTools::interpolate_boundary_values(this->dof_handler,0,
                                 ZeroFunction<dim>(),
                                 boundary_values_adjoint,
                                 this->fe->component_mask(velocities));

  //State
  MatrixTools::apply_boundary_values(boundary_values,
                                   this->LHS.block(2,0),
                                   this->solution.block(2),
                                   this->RHS.block(2));

  //Adjoint
  MatrixTools::apply_boundary_values(boundary_values_adjoint,
                                   this->LHS.block(0,2),
                                   this->solution.block(0),
                                   this->RHS.block(0));

  //Mass matrix
  std::vector<bool> boundary_dofs_vec(this->dof_handler.n_dofs(), false);
  DoFTools::extract_boundary_dofs(this->dof_handler,ComponentMask(),boundary_dofs_vec); //extract boundary dofs
  for (unsigned int i=0; i<this->dof_handler.n_dofs(); i++)
    if (boundary_dofs_vec[i])                                      //if it's a boundary dof
       for (unsigned int j=0; j<this->dof_handler.n_dofs(); j++)   //force the full row to zero
          if (this->LHS.block(0,0).el(i,j)!=0)
             this->LHS.block(0,0).set(i,j,0);


this->LHS.block(2,1).copy_from(this->LHS.block(0,0));
this->LHS.block(2,1)*=-1;


//Second row assembly
 for (int i=0; i<nu; i++){
   this->LHS.block(1,1).set(i,i,this->alpha);
   this->LHS.block(1,2).set(i,i,-1);
   this->LHS.block(1,3).set(i,i,1);
 }

//Identity matrices for dummy control and multiplier
for (unsigned int i=nu; i<this->dof_handler.n_dofs(); i++){
   this->LHS.block(1,1).set(i,i,1);
   this->LHS.block(3,3).set(i,i,1);
 }

this->LHS.block(0,0).vmult_add(this->RHS.block(0),z);

}




template <int dim>
void PDASStokes<dim>::output_result() {
//create .vtk output files and compute norms/errors

 //.vtk file for the state
 std::vector<std::string> solution_names(dim, "velocity");
 solution_names.emplace_back("pressure");
 std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
 data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
 DataOut<dim> data_out;
 data_out.attach_dof_handler(this->dof_handler);

 data_out.add_data_vector(this->solution.block(0),
                           solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);
 data_out.build_patches();
 std::ofstream output("solution_stokes.vtk");
 data_out.write_vtk(output);


  //.vtk file for the control
  std::vector<std::string> solution_names2(dim, "control");
  solution_names2.emplace_back("-");  //dummy control
  DataOut<dim> data_out2;
  data_out2.attach_dof_handler(this->dof_handler);
  data_out2.add_data_vector(this->solution.block(1),
                               solution_names2,
                               DataOut<dim>::type_dof_data,
                               data_component_interpretation);
  data_out2.build_patches();
  std::ofstream output2("solution_control.vtk");
  data_out2.write_vtk(output2);

  //.vtk file for the adjoint
  std::vector<std::string> solution_names3(dim, "adjoint velocity");
  solution_names3.emplace_back("adjoint pressure");
  DataOut<dim> data_out3;
  data_out3.attach_dof_handler(this->dof_handler);
  data_out3.add_data_vector(this->solution.block(2),
                                solution_names3,
                                DataOut<dim>::type_dof_data,
                                data_component_interpretation);
  data_out3.build_patches();
  std::ofstream output3("solution_adjoint.vtk");
  data_out3.write_vtk(output3);


//Errors/norms computation

  //State error:
  //dof_handler for the velocity space only
  int vel_degree=(*(this->fe)).get_sub_fe(0,1).degree;
  FESystem<dim> fe_vel(FE_Q<dim>(vel_degree),dim);
  DoFHandler<dim> dof_handler_vel;
  dof_handler_vel.initialize(this->triangulation,fe_vel);
  DoFRenumbering::Cuthill_McKee(dof_handler_vel,true);
  //extrapolation of the velocity part from the state solution vector
  Vector<double> v_only(dof_handler_vel.n_dofs());
  for (int i=0; i<nu; i++)
    v_only[i]=this->solution.block(0)[i];

  //Error computation for velocity only
  Vector<float> difference_per_cell(this->triangulation.n_active_cells());
  VectorTools::integrate_difference(dof_handler_vel,
                                    v_only,
                                    VectorialFunction<dim>(this->z_expr),
                                    difference_per_cell,
                                    QGauss<dim>(this->fe->degree + 1),
                                    VectorTools::L2_norm);
  const double state_error =
          VectorTools::compute_global_error(this->triangulation,
                                            difference_per_cell,
                                            VectorTools::L2_norm);


 //Control norm
 Vector<float> difference_per_cell2(this->triangulation.n_active_cells());
 VectorTools::integrate_difference(this->dof_handler,
                                  this->solution.block(1),
                                  VectorialFunction<dim>("0,0,0"),
                                  difference_per_cell2,
                                  QGauss<dim>(this->fe->degree + 1),
                                  VectorTools::L2_norm);
  const double u_norm =
                 VectorTools::compute_global_error(this->triangulation,
                                                  difference_per_cell2,
                                                  VectorTools::L2_norm);

 //Adjoint norm
 Vector<float> difference_per_cell3(this->triangulation.n_active_cells());
 VectorTools::integrate_difference(this->dof_handler,
                                  this->solution.block(2),
                                  VectorialFunction<dim>("0,0,0"),
                                  difference_per_cell3,
                                  QGauss<dim>(this->fe->degree + 1),
                                  VectorTools::L2_norm);
 const double p_norm =
                    VectorTools::compute_global_error(this->triangulation,
                                          difference_per_cell3,
                                          VectorTools::L2_norm);


std::cout << "||v-z|| = " << state_error << std::endl;
std::cout << "||u|| = " << u_norm << std::endl;
std::cout << "||p|| = " << p_norm << std::endl;
std::cout << "J = " << 0.5*state_error*state_error + this->alpha/2*u_norm*u_norm<< std::endl;


}
