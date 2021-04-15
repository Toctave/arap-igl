#pragma once

#include "Mesh.hpp"

struct NewtonSolver {
    Eigen::VectorXf rest_points;
    Eigen::SparseMatrix<float> edge_weights;
    Eigen::SparseMatrix<float> laplacian_matrix;
    
    NewtonSolver(const Mesh& mesh);

    void step(Mesh& mesh);
};

