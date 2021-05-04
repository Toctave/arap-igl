#include "gradient_descent_solver.hpp"

GradientDescentSolver::GradientDescentSolver(int max_iterations,
                                             float step_size)
    : max_iterations_(max_iterations),
      step_size_(step_size) {
}

void GradientDescentSolver::improve(EnergyModel &model,
                                    Eigen::VectorXf &guess) {
    model.set_query_point(guess);
    guess -= step_size_ * model.gradient();
}

void GradientDescentSolver::solve(EnergyModel &model, Eigen::VectorXf &guess) {
    for (int i = 0; i < max_iterations_; i++) {
        improve(model, guess);
    }
}
