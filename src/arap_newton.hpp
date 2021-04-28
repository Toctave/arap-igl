#pragma once

#include "Mesh.hpp"

class NewtonSolver {
private:
    const Eigen::VectorXf rest_points_;
    const Eigen::SparseMatrix<float> edge_weights_;
    const Eigen::SparseMatrix<float> laplacian_matrix_;
    
    Eigen::VectorXf current_points_;

    mutable Eigen::Matrix<float, Eigen::Dynamic, 3> rotations_;
    mutable bool rotations_cached_;

    float step_size_;
    
    void cache_rotations() const;

public:
    NewtonSolver(const Mesh& mesh);

    void apply(Mesh& mesh) const;

    void set_points(const Eigen::VectorXf& points);
    void set_points(const Mesh& mesh);
    
    void step();
    void solve();
    
    float energy() const;
    
    Eigen::VectorXf empirical_gradient();
    Eigen::VectorXf gradient() const;
    
    Eigen::SparseMatrix<float> hessian() const;
    Eigen::SparseMatrix<float> empirical_hessian();
    
    
    // float trust_radius;
    // float max_trust_radius;

    // float increase_factor;
    // float decrease_factor;

    // float increase_threshold;
    // float decrease_threshold;

    // float accept_threshold;
};

