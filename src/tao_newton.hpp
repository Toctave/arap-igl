#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <petsctao.h>

class TaoSolver {
private:
    Tao m_tao;

public:
    TaoSolver(Tao tao);
};

int tao_test_main(int argc,char **argv);

