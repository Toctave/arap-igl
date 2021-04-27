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

struct FixedVertex {
    Eigen::Index index;
    size_t group;
};

struct LaplacianSystem {
    Mesh* mesh;
    Points V0;
    
    Eigen::SparseMatrix<float> edge_weights;
    std::vector<int> fixed_vertex_indices;

    std::vector<Eigen::Matrix3f> optimal_rotations;
    float rotation_variation_penalty;
    
    Eigen::SparseMatrix<float> laplacian_matrix;
    Eigen::Matrix<float, Eigen::Dynamic, 3> rhs;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;

    int iterations;

    void set_alpha(float alpha);
};

void system_init(LaplacianSystem& system, Mesh* mesh, float alpha);
bool system_bind(LaplacianSystem& system,
		 const Points& V0,
		 const std::vector<FixedVertex>& fixed_vertices,
		 WeightType type);
void system_solve(LaplacianSystem& system, int iterations);
bool system_iterate(LaplacianSystem& system);

float system_energy(const LaplacianSystem& system);

Eigen::SparseMatrix<float> uniform_weights(const Mesh& mesh);
Eigen::SparseMatrix<float> cotangent_weights(const Mesh& mesh);
Eigen::SparseMatrix<float> cotangent_weights_abs(const Mesh& mesh);
Eigen::SparseMatrix<float> cotangent_weights_clamped(const Mesh& mesh);
Eigen::SparseMatrix<float> mean_value_weights(const Mesh& mesh);


