/********************************************************************************
PDASProblem Class: abstract base class for generic OCPs solved with PDAS method
********************************************************************************/
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_in.h>

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
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include "muParser.h"

using namespace dealii;
using namespace mu;


//Class to handle vectorial functions
template <int dim>
class VectorialFunction : public Function<dim>
{
public:
  VectorialFunction(std::string p)
    : Function<dim>(dim),
    expr(p)
  {  }
  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &  value) const override;

  private:
    const std::string expr;  //string with the mathematical expression of the function

};


template <int dim>
void VectorialFunction<dim>::vector_value(const Point<dim> &p,
                                       Vector<double> &  values) const
{ //Parser initialization
  Parser parser;
  double a=p(0);
  double b=p(1);
  double c=p(2);
  parser.SetExpr(expr);
  parser.DefineVar("x", &a);
  parser.DefineVar("y", &b);
  parser.DefineVar("z", &c);

  //function evaluation
try{
  int nNum;
  value_type *v = parser.Eval(nNum);
  for (int i=0; i<nNum; ++i)
  {
   values[i]= v[i];
  }
}
catch (Parser::exception_type &excp)
{
std::cout << excp.GetMsg() << std::endl;
}

}


//Class to handle scalar functions
template <int dim>
class ScalarFunction : public Function<dim>
{
public:
  ScalarFunction(std::string p)
    : Function<dim>(1),
    expr(p)
  {  }
  virtual double value(const Point<dim> & p,
                       const unsigned int component=0) const override;

  private:
    std::string expr;  //string with the mathematical expression of the function


};
template <int dim>
double ScalarFunction<dim>::value(const Point<dim> & p,
                                 const unsigned int component ) const
{
  Assert(component < this->n_components,
         ExcIndexRange(component, 0, this->n_components));

 //Parser initialization
  Parser parser;
  double a=p(0);
  double b=p(1);
  double c=p(2);
  parser.SetExpr(expr);
  parser.DefineVar("x", &a);
  parser.DefineVar("y", &b);
  parser.DefineVar("z", &c);

  //function evaluation
  double val=0;
  try { val=parser.Eval();}
  catch (Parser::exception_type &excp)
    {std::cout << excp.GetMsg() << std::endl;}

 return  val;
}




template <int dim>
class PDASProblem {
public:
  PDASProblem(std::string file_, int problem_type);
  void run();  // run PDAS algorithm

protected:
  void make_grid();                                      //prepare the grid
  void setup_system();                                   //initialize dof_handler, matrices and vectors
  virtual void dof_renumbering()=0;                      //dof renumbering
  virtual void assemble_system()=0;                      //assemble all blocks that do not depend on the iteration
  void assemble_system_update();                         //assemble all blocks that depend on the iteration
  void solve_system();                                   //solve LHS*solution=RHS
  void check_convergence();                              //check convergence for PDAS
  virtual void output_result()=0;                        //create output files and compute errors/norms
  void find_set();                                       //identify the active and inactive sets

  Triangulation<dim> triangulation;                      //handles the mesh
  std::unique_ptr<FESystem<dim>> fe;                     //handles the finite element space
  DoFHandler<dim> dof_handler;                           //handles the dofs
  BlockSparsityPattern  block_sparsity_pattern;          //sparsity pattern of LHS matrix

  BlockSparseMatrix<double> LHS;                         //LHS matrix
  BlockVector<double> solution;                          //solution vector
  BlockVector<double> RHS;                               //RHS matrix

  std::string mesh;                                      //name of the .msh file containing the mesh
  std::string f_expr;                                    //mathematical expression of the forcing term
  std::string z_expr;                                    //mathematical expression of the desired state
  std::string bc_expr;                                   //mathematical expression of the boundary conditions
  std::string a_expr;                                    //mathematical expression of the lower bound for the control
  std::string b_expr;                                    //mathematical expression of the upper bound for the control

  Vector<double> umax;                                   //values of the upper bound for the control in the nodes
  Vector<double> umin;                                   //values of the lower bound for the control in the nodes
  double alpha;                                          //penalization parameter for the control norm in the cost functional
  bool convergence=false;
  int itmax;                                             //maximum number of iterations allowed for PDAS

  Vector<int> set;                                       //assign at each node the corresponding active/inactive set
  Vector<int> set0;                                      //values of set at the previous iteration
};

template <int dim>
PDASProblem<dim>::PDASProblem(std::string file_, int problem_type)
  {
    ParameterHandler parameters;

    //Parameters declarations
    parameters.enter_subsection("Geometry");
      parameters.declare_entry("Mesh File", " ");
    parameters.leave_subsection();
    parameters.enter_subsection("Problem");
      parameters.declare_entry("forcing term", " ");
      parameters.declare_entry("boundary conditions", " ");
    parameters.leave_subsection();
    parameters.enter_subsection("PDAS");
      parameters.declare_entry("alpha", "0", Patterns::Double(0));
      parameters.declare_entry("kmax", "0", Patterns::Integer(0));
      parameters.declare_entry("a", " ");
      parameters.declare_entry("b", " ");
      parameters.declare_entry("z", " ");
  parameters.leave_subsection();
  parameters.declare_entry("Fe","1",Patterns::Integer(1));

    //Read the input parameters
    parameters.parse_input(file_);
    parameters.enter_subsection("Geometry");
      mesh=parameters.get("Mesh File");
    parameters.leave_subsection();
    parameters.enter_subsection("Problem");
      f_expr=parameters.get("forcing term");
      bc_expr=parameters.get("boundary conditions");
    parameters.leave_subsection();
    parameters.enter_subsection("PDAS");
      alpha=parameters.get_double("alpha");
      itmax=parameters.get_integer("kmax");
      a_expr=parameters.get("a");
      b_expr=parameters.get("b");
      z_expr=parameters.get("z");
    parameters.leave_subsection();
    int fe_order=parameters.get_integer("Fe");

    if (problem_type==1) //Laplace
       fe=std::make_unique<FESystem<dim>>(FE_Q<dim>(fe_order));
    else if (problem_type==2) //Stokes
       fe=std::make_unique<FESystem<dim>>(FE_Q<dim>(fe_order+1),dim,FE_Q<dim>(fe_order),1);
  }


template <int dim>
void PDASProblem<dim>::make_grid(){

  GridIn<dim> gridin;
  gridin.attach_triangulation(triangulation);
  std::ifstream f(mesh);
  gridin.read_msh(f);  //read mesh from file

  //force all boundary tags to zero to be consistent with apply_boundary_values function
  for (const auto &cell : triangulation.active_cell_iterators())
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        if (cell->face(f)->at_boundary()){
         cell->face(f)->set_all_boundary_ids(0);}

  std::cout << "Number of cells : " << triangulation.n_active_cells() << std::endl;

}


template <int dim>
void PDASProblem<dim>::setup_system(){

    //Dof_handler initialization

      dof_handler.initialize(triangulation,*fe);
      std::cout <<"Number of DOFs: " << dof_handler.n_dofs() << std::endl;
      dof_renumbering();

    //Sparsity pattern creation:

      //sparsity pattern for state/mass/adjoint blocks
      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      SparsityPattern sparsity_pattern;
      DoFTools::make_sparsity_pattern(dof_handler, dsp);
      sparsity_pattern.copy_from(dsp);

       //sparsity pattern for diagonal blocks
      SparsityPattern diagonal_sparsity_pattern(dof_handler.n_dofs(),1);

      //Full sparsity pattern
      block_sparsity_pattern.reinit(4,4);
      block_sparsity_pattern.block(0,0).copy_from(sparsity_pattern);
      block_sparsity_pattern.block(0,2).copy_from(sparsity_pattern);
      block_sparsity_pattern.block(2,0).copy_from(sparsity_pattern);
      block_sparsity_pattern.block(2,1).copy_from(sparsity_pattern);
      block_sparsity_pattern.block(1,1).copy_from(diagonal_sparsity_pattern);
      block_sparsity_pattern.block(1,2).copy_from(diagonal_sparsity_pattern);
      block_sparsity_pattern.block(1,3).copy_from(diagonal_sparsity_pattern);
      block_sparsity_pattern.block(3,1).copy_from(diagonal_sparsity_pattern);
      block_sparsity_pattern.block(3,3).copy_from(diagonal_sparsity_pattern);
      block_sparsity_pattern.block(0,1).copy_from(diagonal_sparsity_pattern);
      block_sparsity_pattern.block(0,3).copy_from(diagonal_sparsity_pattern);
      block_sparsity_pattern.block(1,0).copy_from(diagonal_sparsity_pattern);
      block_sparsity_pattern.block(2,2).copy_from(diagonal_sparsity_pattern);
      block_sparsity_pattern.block(2,3).copy_from(diagonal_sparsity_pattern);
      block_sparsity_pattern.block(3,0).copy_from(diagonal_sparsity_pattern);
      block_sparsity_pattern.block(3,2).copy_from(diagonal_sparsity_pattern);

      block_sparsity_pattern.collect_sizes();

    //Vectors and matrixes initialization

      LHS.reinit(block_sparsity_pattern);
      RHS.reinit(4,dof_handler.n_dofs());
      solution.reinit(4,dof_handler.n_dofs());

      set0.reinit(dof_handler.n_dofs());
      set.reinit(dof_handler.n_dofs());

      umax.reinit(dof_handler.n_dofs());
      umin.reinit(dof_handler.n_dofs());
}



template <int dim>
void PDASProblem<dim>::assemble_system_update(){
//assemble the blocks of LHS and RHS that change at each iteration of the optimization cycle

        for (unsigned int i=0; i<dof_handler.n_dofs();i++)
            if (set0[i]==1){   //active set A+
                LHS.block(3,1).set(i,i,1);
                LHS.block(3,3).set(i,i,0);
                RHS.block(3)[i]=umax[i];
              }
              else if (set0[i]==2){ //active set A-
                LHS.block(3,1).set(i,i,1);
                LHS.block(3,3).set(i,i,0);
                RHS.block(3)[i]=umin[i];
                }
              else{ //inactive set I
                LHS.block(3,1).set(i,i,0);
                LHS.block(3,3).set(i,i,1);
                RHS.block(3)[i]=0;
                }
}


template <int dim>
void PDASProblem<dim>::solve_system(){
//solve LHS*solution=RHS

  SparseDirectUMFPACK mat;
  mat.initialize(LHS); //LU factorization of LHS
  mat.vmult(solution,RHS);
}


template <int dim>
void PDASProblem<dim>::find_set(){
//associate to each node the corresponding active/inactive set

  for (unsigned int i=0; i<dof_handler.n_dofs(); i++)
    if (solution.block(1)[i]+solution.block(3)[i]-umax[i] > 0)
      set[i]=1;  //active set A+
      else if (solution.block(1)[i]+solution.block(3)[i]-umin[i] < 0)
        set[i]=2; //active set A-
        else
          set[i]=3;  //inactive set I
}


template <int dim>
void PDASProblem<dim>::check_convergence() {
//check convergence of the PDAS algorithm
   if (set0==set){
     std::cout << "Convergence reached" << std::endl;
     convergence=true;
   }
}


template <int dim>
void PDASProblem<dim>::run() {
//run the PDAS algorithm

  std::cout << "Solving problem in " << dim << "D" << std::endl << std::endl;

  make_grid();
  setup_system();
  assemble_system();

  find_set();
  set0=set;

  int k=1;

  while (k<itmax && convergence==false){


    std::cout << std::endl << "Iteration " << k << std::endl;

    assemble_system_update();

    solve_system();

    find_set();
    check_convergence();

    output_result();
    //Updates for the next iteration:
    set0=set;
    k++;
  }
}
