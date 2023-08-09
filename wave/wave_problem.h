#ifndef WAVE_PROBLEM_H
#define WAVE_PROBLEM_H


#include <deal.II/lac/sparse_direct.h>
#include <deal.II/base/time_stepping.h>
#include <cmath>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h> //The class function is used for defining the boundary condition and rhs
#include <deal.II/base/discrete_time.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_richardson.h>

#include <deal.II/lac/precondition_block.h>
#include <deal.II/numerics/derivative_approximation.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <iostream>
#include <fstream>

namespace wave
{

using namespace dealii;

template<int dim>
class InitialValuesQ : public Function<dim>
{
	public:
		virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
		{
			double x = p[0];
			return std::exp(-64.0*x*x);
		}
};

template<int dim>
class parameterReader
{
	public:
		parameterReader(ParameterHandler &);
		void read_parameters(const std::string &);
	private:
		void declare_parameters();
		ParameterHandler &prm;
};

template<int dim>
parameterReader<dim>::parameterReader(ParameterHandler &paramHandler)
:prm(paramHandler)
{}

template<int dim>
void parameterReader<dim>::read_parameters(const std::string &parameter_file)
{
	declare_parameters();
	prm.parse_input(parameter_file);
}

template<int dim>
void parameterReader<dim>::declare_parameters()
{
	prm.enter_subsection("geometry");
	{
		prm.declare_entry("refinement level",
		    			"6",
		    			Patterns::Integer(0,10),
		    			"Refinement level of the mesh");
		prm.declare_entry("left end",
		    			"-0.5",
		    			Patterns::Double(),
					"Coordinates of the left edge");
		prm.declare_entry("right end",
		    			"0.5",
		    			Patterns::Double(0),
					"Coordinates of right edge");
	}
	prm.leave_subsection();

	prm.enter_subsection("finite element");
	{

	prm.declare_entry("order of polynomial",
				"2",
				Patterns::Integer(0,10),
		   "polynomial degree of DG elements ");
	}
	prm.leave_subsection();

	prm.enter_subsection("solution methods");
	{
		prm.declare_entry("integration method",
		    			"RK_THIRD_ORDER",
		    			Patterns::Anything(),
		    			"Intergration method used");
		prm.declare_entry("cfl number",
		    			"1.0",
		    			Patterns::Double(),
		    			"CFL Number");
	}
	prm.leave_subsection();
}


template <int dim>
class waveProblem
{
public:
	waveProblem(ParameterHandler &,int);
	void run();
private:
	void make_grid();
	void setup_system();
	void assemble_system();
	void solve();
	void explicit_method(const TimeStepping::runge_kutta_method method);
	void output_results();
	void output_data(const double,const uint,TimeStepping::runge_kutta_method)const;
	Vector<double> right_hand_side(const double,const Vector<double> &)const;
	double error_calc();
	double CFL_calc();


	

	Triangulation<dim> triangulation;
	FE_DGQ<dim> fe;
	DoFHandler<dim> dof_handler;

	SparsityPattern		sparsity_pattern;
	SparseMatrix<double>	system_mass_matrix;
//	SparseMatrix<double>	system_diff_matrix;


//	FullMatrix<double>	system_mass_matrix;
	FullMatrix<double>	system_diff_matrix;
	FullMatrix<double>	system_flux_matrix;
	FullMatrix<double>	sum_matrix;
	SparseMatrix<double>	stiffness_matrix;

	SparseDirectUMFPACK mass_inverse;	

	Vector<double>	solution;
	Vector<double>	flux;
	Vector<double>	old_solution;
	Vector<double>	old_flux;
	Vector<double>	system_rhs;

	int no_dofs;
	double CFL;

	//const QGauss<dim>	quadrature;
	//const QGauss<dim - 1> quadrature_face;
	double	time, time_step, stop_time, initial_time;
	int refinement_level;
	ParameterHandler &prm;
};

template <int dim>
waveProblem<dim>::waveProblem(ParameterHandler &param,int order)
	:prm(param),
	fe(order) // Order of the polynomial
	,dof_handler(triangulation)
{
	prm.enter_subsection("geometry");
	refinement_level=prm.get_integer("refinement level");
	prm.leave_subsection();
	prm.enter_subsection("solution methods");
	CFL=prm.get_double("cfl number");
	prm.leave_subsection();
}

template <int dim>
void waveProblem<dim>::make_grid()
{
	double right,left;
	prm.enter_subsection("geometry");
	left=prm.get_double("left end");
	right=prm.get_double("right end");
	prm.leave_subsection();
	GridGenerator::hyper_cube(triangulation, left ,right);
	triangulation.refine_global(refinement_level);

}

template<int dim>
void waveProblem<dim>::setup_system()
{
	dof_handler.distribute_dofs(fe);
	no_dofs=dof_handler.n_dofs();


	DynamicSparsityPattern dsp(dof_handler.n_dofs(),
				dof_handler.n_dofs());
	DoFTools::make_flux_sparsity_pattern(dof_handler,dsp);
	sparsity_pattern.copy_from(dsp);

	system_mass_matrix.reinit(sparsity_pattern);
/*	system_diff_matrix.reinit(sparsity_pattern);
//	system_flux_matrix.reinit(sparsity_pattern);
	sum_matrix.reinit(sparsity_pattern);
	stiffness_matrix.reinit(sparsity_pattern);
*/
	solution.reinit(dof_handler.n_dofs());
	old_solution.reinit(dof_handler.n_dofs());
	flux.reinit(dof_handler.n_dofs());
	old_flux.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());

	std::cout<<"Degrees of freedom : "<<no_dofs<<std::endl;

//	system_mass_matrix.reinit(no_dofs,no_dofs);
	system_diff_matrix.reinit(no_dofs,no_dofs);
	system_flux_matrix.reinit(no_dofs,no_dofs);
	sum_matrix.reinit(no_dofs,no_dofs);
}

template<int dim>
void waveProblem<dim>::assemble_system()
{
	QGaussLobatto<dim>	quadrature_formula(fe.get_degree() + 2);
	QGaussLobatto<dim-1>	face_quadrature_formula(fe.get_degree() + 2);

	const unsigned int n_q_points = quadrature_formula.size();
	const unsigned int n_face_q_points = face_quadrature_formula.size();

	const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

	FullMatrix<double>	mass_matrix(dofs_per_cell, dofs_per_cell);
	FullMatrix<double>	diff_matrix(dofs_per_cell, dofs_per_cell);
	FullMatrix<double>	flux_matrix(dofs_per_cell, dofs_per_cell);
	Vector<double>		cell_rhs(dofs_per_cell);
	
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	FEValues<dim>		fe_values(fe,
			  	quadrature_formula,
			  	update_values | update_gradients |
			  	update_quadrature_points | update_JxW_values);
	FEFaceValues<dim> 	fe_face_values(fe,
			  	face_quadrature_formula,
			  	update_values | update_gradients |
			  	update_quadrature_points | update_JxW_values);
	
	for (const auto &cell : dof_handler.active_cell_iterators())
	{
		mass_matrix = 0;
		diff_matrix = 0;
		flux_matrix = 0;
      		cell_rhs = 0;

		fe_values.reinit(cell);

		for (unsigned int q_points = 0; q_points < n_q_points; ++q_points)
       			for(unsigned int i = 0; i < dofs_per_cell; ++i)
			{
				for(unsigned int j = 0; j < dofs_per_cell; ++j)
				{
					mass_matrix(i,j) +=
      						fe_values.shape_value(i,q_points) *
      						fe_values.shape_value(j,q_points) *
      						fe_values.JxW(q_points);
					

					diff_matrix(i,j) +=	  
						fe_values.shape_grad(i, q_points)[0] *
                                                fe_values.shape_value(j, q_points) *
                                                fe_values.JxW(q_points);

				}
			}
//Flux Caluculation		
		typename DoFHandler<dim>::active_cell_iterator temp = dof_handler.begin_active();
		typename DoFHandler<dim>::active_cell_iterator copy = dof_handler.begin_active();
		typename DoFHandler<dim>::active_cell_iterator last = dof_handler.begin_active();
		int left_norm=-1,right_norm=1,delta=0;
		for(;copy!=dof_handler.end();++copy)
		{last=copy;}
		uint global_i,global_j;
		for( auto cell=dof_handler.begin_active();
      			cell!=dof_handler.end();
      			cell++)
		{
			uint n_faces=GeometryInfo<dim>::faces_per_cell;
			std::vector<types::global_dof_index>	dof_indices(dofs_per_cell);
			std::vector<types::global_dof_index>	temp_dof_indices(dofs_per_cell);
			cell->get_dof_indices(dof_indices);
			if(cell==dof_handler.begin_active())
			{temp=last;}
			else
			{temp=cell;
			temp--;}
			temp->get_dof_indices(temp_dof_indices);
			global_i=dof_indices[0];
			global_j=temp_dof_indices[dofs_per_cell-1];
			system_flux_matrix(global_i,global_i)=0.5*left_norm*(1+(delta*left_norm));
			system_flux_matrix(global_i,global_j)=0.5*left_norm*(1-(delta*left_norm));
			if(cell==last)
			{temp=dof_handler.begin_active();}
			else
			{temp=cell;temp++;}
			temp->get_dof_indices(temp_dof_indices);
			global_i=dof_indices[dofs_per_cell-1];
			global_j=temp_dof_indices[0];
			system_flux_matrix(global_i,global_i)=0.5*right_norm*(1+(delta*right_norm));
			system_flux_matrix(global_i,global_j)=0.5*right_norm*(1-(delta*right_norm));
		}

		cell->get_dof_indices(local_dof_indices);
		for (unsigned int i =0; i < dofs_per_cell; ++i)
		{
			for (unsigned int j = 0; j < dofs_per_cell; ++j)
			{
				system_mass_matrix.add(local_dof_indices[i],local_dof_indices[j],mass_matrix(i,j));
				system_diff_matrix.add(local_dof_indices[i],local_dof_indices[j],diff_matrix(i,j));
			}
		}

	}	
	mass_inverse.initialize(system_mass_matrix);
	//std::cout << "Number of quadrature points on the face : " << n_face_q_points << std::endl;
	//std::cout << "Total number of qudrature points : " << n_q_points << std::endl;
}
template<int dim>
void waveProblem<dim>::solve()
{

	InitialValuesQ<dim> initial_values;
	VectorTools::interpolate(dof_handler, initial_values, solution);
	sum_matrix.add(1.0,system_diff_matrix);
	sum_matrix.add(-1.0,system_flux_matrix);
	right_hand_side(0,solution);

	std::string solution_methods;
	prm.enter_subsection("solution methods");
	if(prm.get("integration method")=="RK_THIRD_ORDER")
	{
		explicit_method(TimeStepping::RK_THIRD_ORDER);
	}
	
}
template<int dim>
Vector<double> waveProblem<dim>::right_hand_side(const double time,const Vector<double> &y)const
{
	Vector<double> temp(dof_handler.n_dofs());
	Vector<double> value(dof_handler.n_dofs());
	temp=0;
	sum_matrix.vmult(temp,y);
	
	mass_inverse.vmult(value,temp);

/*	std::cout<<"f(q,t) : "<<std::endl;
	for(uint i=0;i<dof_handler.n_dofs();i++)
		std::cout<<value[i]<<'\n';
*/	return value;

}

template<int dim>
void waveProblem<dim>::explicit_method(const TimeStepping::runge_kutta_method method)
{
	initial_time 	= 0;
	stop_time	= 1;
//	int steps 	= 5000;
//	time_step	=(stop_time-initial_time)/steps;
	time_step 	= CFL_calc(); 
	std::cout<<"time step size : "<<time_step<<std::endl;


	//Initialisation
	InitialValuesQ<dim> initial_values;
	//Initialises solution with initial_values
	VectorTools::interpolate(dof_handler, initial_values, solution);
	
	//TimeStepping method selection
	TimeStepping::ExplicitRungeKutta<Vector<double>>	explicit_runge_kutta(method);
	output_data(initial_time,0,method);

	DiscreteTime 	time(initial_time,stop_time,time_step);

	double temp_solution;
	while(time.is_at_end()==false)
	{
		temp_solution=solution[dof_handler.n_dofs()-1];
		solution[0]=temp_solution;
		explicit_runge_kutta.evolve_one_time_step(
		[this](const double func_time, const Vector<double> &y)
			{return this->right_hand_side(func_time,y);},
		time.get_current_time(),
		time.get_next_step_size(),
		solution);
		time.advance_time();
		
		std::cout<<"\r Time : "<<time.get_current_time()<<std::flush;

		if(static_cast<int>(time.get_current_time()) % 10==0)
		{
			output_data(time.get_current_time(),time.get_step_number(),method);
		}

	}
	std::cout<<'\n';
	output_results();

}

template<int dim>
double waveProblem<dim>::error_calc()
{
	Vector<double> exact_solution(dof_handler.n_dofs());

	VectorTools::integrate_difference(dof_handler,
				   	solution,
				   	InitialValuesQ<dim>(),
				   	exact_solution,
				   	QGaussLobatto<dim>(fe.get_degree()+2),VectorTools::L2_norm);
	double l2_norm = std::sqrt(exact_solution.l2_norm());

	return l2_norm;
}

template<int dim>
double waveProblem<dim>::CFL_calc()
{
	double h=0,
		delta_t,
		u=1,
		factor=1;
	
	
	int poly_order = fe.get_degree();
	factor/=3;
	for(const auto cell:triangulation.active_cell_iterators())
	{
		h=std::max(h,cell->diameter());
	}
	std::cout<<"mix h : "<<h<<std::endl;
	delta_t=CFL*h*factor/(u*(2*poly_order+1));
	return delta_t;
}

template<int dim>
void waveProblem<dim>::output_data(const double time, const uint time_step, TimeStepping::runge_kutta_method method)const
{
	std::string 	method_name;
	method_name="RK4";

	DataOut<dim> data_out;
	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(solution,"solution");

	data_out.build_patches();

	data_out.set_flags(DataOutBase::VtkFlags(time,time_step));

	const std::string filename = "solution_"+method_name+"_"+Utilities::int_to_string(time_step,3)+Utilities::int_to_string(time,3)+"_"+".vtk";

	std::ofstream output(filename);
	data_out.write_vtk(output);
}

template<int dim>
void waveProblem<dim>::output_results()
{
/*	std::cout<<"Solution : "<<std::endl;
	for(uint i=0;i<dof_handler.n_dofs();++i)
	{
		std::cout<<solution[i]<<std::endl;
	}
*/	std::cout<<"L2 norm : "<<error_calc()<<std::endl;
}

template<int dim>
void waveProblem<dim>::run()
{
	make_grid();
	setup_system();
	assemble_system();
	solve();
}

}//namespace of wave
//#include "wave_problem.cc"
#endif
