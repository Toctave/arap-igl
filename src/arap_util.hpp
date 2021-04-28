#pragma once
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

#include "Mesh.hpp"

enum WeightType {
    UNIFORM,
    COTANGENT,
    COTANGENT_CLAMPED,
    COTANGENT_ABS,
    MEAN_VALUE,
    WEIGHT_TYPE_MAX,
};

Eigen::SparseMatrix<float> uniform_weights(const Mesh& mesh);
Eigen::SparseMatrix<float> cotangent_weights(const Mesh& mesh);
Eigen::SparseMatrix<float> cotangent_weights_abs(const Mesh& mesh);
Eigen::SparseMatrix<float> cotangent_weights_clamped(const Mesh& mesh);
Eigen::SparseMatrix<float> mean_value_weights(const Mesh& mesh);

Eigen::Matrix<float, Eigen::Dynamic, 3>
compute_best_rotations(const Eigen::VectorXf& points,
		       const Eigen::VectorXf& rest_points,
		       const Eigen::SparseMatrix<float>& edge_weights);
Eigen::SparseMatrix<float> compute_laplacian_matrix(const Eigen::SparseMatrix<float>& edge_weights);
