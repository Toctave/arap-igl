#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

class EnergyModel {
public:
    virtual void set_query_point(const Eigen::VectorXf& x) = 0;
    
    virtual float energy() const = 0;
    virtual Eigen::VectorXf gradient() const = 0;
    virtual Eigen::SparseMatrix<float> hessian() const = 0;
    
    virtual int degrees_of_freedom() const = 0;
};

class EnergySolver {
public:
    virtual void improve(EnergyModel& model, Eigen::VectorXf& guess) = 0;
    virtual void solve(EnergyModel& model, Eigen::VectorXf& guess) = 0;
};
