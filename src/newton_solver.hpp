#pragma once

#include "model_solver.hpp"

class NewtonSolver {
private:
    int max_iterations_;
    float step_size_;
    
public:
    NewtonSolver(int max_iterations, float step_size);
    
    virtual void improve(EnergyModel& model, Eigen::VectorXf& guess);
    virtual void solve(EnergyModel& model, Eigen::VectorXf& guess);
};
