#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <petsctao.h>

#include "arap_newton.hpp"

class TaoSolver {
private:
    Tao m_tao;

public:
    TaoSolver(Tao tao);
};

Tao tao_init();
void tao_finalize(Tao tao);
void tao_solve_newton(Tao tao, NewtonSolver& solver);

