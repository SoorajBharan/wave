#include "wave_problem.h"
#include <iostream>

int main()
{
	std::string input_file = "input_parameters.prm";
	wave::ParameterHandler prm;
	wave::parameterReader<1> param(prm);
	param.read_parameters("input.prm");

	prm.print_parameters(std::cout,dealii::ParameterHandler::Text);
	
	int order_of_poly;
	prm.enter_subsection("finite element");
	order_of_poly = prm.get_integer("order of polynomial");
	prm.leave_subsection();
	wave::waveProblem<1>	object(prm,order_of_poly);
	object.run();
	
	// dealii::ParameterHandler prm;
	// prm.enter_subsection("Geometry");
	// {
	// 	prm.declare_entry("refinement_level",
	// 				"4",
	// 				dealii::Patterns::Integer(0,10));
	// }
	// prm.leave_subsection();
	//
	// prm.parse_input("input.prm");
	//
	// prm.enter_subsection("Geometry");
	// std::cout<<"Refinement level : "<<prm.get_integer("refinement_level")<<std::endl;
	// prm.leave_subsection();


	return 0;
}
