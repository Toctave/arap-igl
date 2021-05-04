#pragma once

#include "model_solver.hpp"

class GradientDescentSolver {
private:
    int max_iterations_;
    float step_size_;
    
public:
    GradientDescentSolver(int max_iterations, float step_size);
    
    virtual void improve(EnergyModel& model, Eigen::VectorXf& guess);
    virtual void solve(EnergyModel& model, Eigen::VectorXf& guess);
};
