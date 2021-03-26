#pragma once
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

struct Mesh {
    Eigen::MatrixXf V;
    Eigen::MatrixXi F;
};

enum WeightType {
    COTANGENT,
    COTANGENT_CLAMPED,
    COTANGENT_ABS,
    MEAN_VALUE,
};

struct FixedVertex {
    Eigen::Index index;
    size_t group;
};

struct LaplacianSystem {
    Mesh* mesh;
    Eigen::MatrixXf V0;
    Eigen::SparseMatrix<float> edge_weights;
    std::vector<Eigen::Index> fixed_vertex_indices;

    std::vector<Eigen::Matrix3f> optimal_rotations;
    float rotation_variation_penalty;
    
    Eigen::SparseMatrix<float> laplacian_matrix;
    Eigen::Matrix<float, Eigen::Dynamic, 3> rhs;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;

    int iterations;
};

void system_init(LaplacianSystem& system, Mesh* mesh, float alpha);
bool system_bind(LaplacianSystem& system,
		 const std::vector<FixedVertex>& fixed_vertices,
		 WeightType type);
void system_solve(LaplacianSystem& system, int iterations);
bool system_iterate(LaplacianSystem& system);
