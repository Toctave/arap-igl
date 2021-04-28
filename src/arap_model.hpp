#pragma once

#include "Mesh.hpp"
#include "model_solver.hpp"

class ARAPModel : public EnergyModel {
    const Eigen::VectorXf rest_points_;
    const Eigen::SparseMatrix<float> edge_weights_;
    const Eigen::SparseMatrix<float> laplacian_matrix_;

    Eigen::VectorXf current_points_;
    Eigen::Matrix<float, Eigen::Dynamic, 3> rotations_;
    
public:
    ARAPModel(const Mesh& mesh);
    
    virtual void set_query_point(const Eigen::VectorXf& x) override;
    
    virtual float energy() const override;
    virtual Eigen::VectorXf gradient() const override;
    virtual Eigen::SparseMatrix<float> hessian() const override;

    virtual int degrees_of_freedom() const;
};
