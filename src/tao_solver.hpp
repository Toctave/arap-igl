#pragma once

#include <petsctao.h>

#include "model_solver.hpp"

class TaoSolver : public EnergySolver {
private:
    Tao tao_;
    Vec tao_points_;
    Mat tao_hessian_;

    TaoSolver(const TaoSolver&);

public:
    TaoSolver();
    ~TaoSolver();
    
    void improve(EnergyModel& model, Eigen::VectorXf& guess);
    void solve(EnergyModel& model, Eigen::VectorXf& guess);
};
