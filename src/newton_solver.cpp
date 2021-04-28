#include "newton_solver.hpp"

#include <Eigen/SparseCholesky>

NewtonSolver::NewtonSolver(int max_iterations, float step_size)
    : max_iterations_(max_iterations),
      step_size_(step_size) {
}
    
void NewtonSolver::improve(EnergyModel& model, Eigen::VectorXf& guess) {
    model.set_query_point(guess);
    Eigen::VectorXf gradient = model.gradient();
    Eigen::SparseMatrix<float> hessian = model.hessian();

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver(hessian);

    if (solver.info() != Eigen::Success) {
	throw std::runtime_error("Could not factorize Newton hessian");
    }

    Eigen::VectorXf delta =
	solver.solve(gradient);

    if (solver.info() != Eigen::Success) {
	throw std::runtime_error("Could not invert Newton hessian");
    }
    
    guess -= step_size_ * delta;
}
    
void NewtonSolver::solve(EnergyModel& model, Eigen::VectorXf& guess) {
    for (int i = 0; i < max_iterations_; i++) {
	improve(model, guess);
    }
}
