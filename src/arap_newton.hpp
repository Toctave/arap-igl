#pragma once

#include "Mesh.hpp"

class NewtonSolver {
private:
    const Eigen::VectorXf rest_points_;
    const Eigen::SparseMatrix<float> edge_weights_;
    const Eigen::SparseMatrix<float> laplacian_matrix_;
    
    Eigen::VectorXf current_points_;

    Eigen::Matrix<float, Eigen::Dynamic, 3> rotations_;
    float step_size_;

    // void compute_best_rotations();

public:
    NewtonSolver(const Mesh& mesh);

    void apply(Mesh& mesh) const;

    void set_points(const Eigen::VectorXf& points);
    void set_points(const Mesh& mesh);
    
    void step();
    void solve();
    
    float energy() const;
    Eigen::VectorXf gradient() const;
    Eigen::SparseMatrix<float> hessian() const;
    
    // float trust_radius;
    // float max_trust_radius;

    // float increase_factor;
    // float decrease_factor;

    // float increase_threshold;
    // float decrease_threshold;

    // float accept_threshold;
};

