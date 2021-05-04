#pragma once

#include "Mesh.hpp"
#include "model_solver.hpp"

class VolumeARAPModel : public EnergyModel {
    // pre-computed :
    TetraMesh mesh_;
    Eigen::Matrix<float, Eigen::Dynamic, 6, Eigen::RowMajor> edge_weights_;
    Eigen::VectorXf volumes_;
    Eigen::Matrix<float, 3, Eigen::Dynamic> volume_gradients_;
    Eigen::Matrix<float, 3, Eigen::Dynamic> curls_;

    float alpha2_;
    float beta2_;

    // Depends on current configuration :
    Points current_points_;
    Eigen::Matrix<float, 3, Eigen::Dynamic> rotations_;
    Eigen::Matrix<float, 3, Eigen::Dynamic> rotations_y_;

    // Helpers :
    Eigen::Ref<Eigen::Matrix3f> curl(Eigen::Index tet, int idx);
    Eigen::Ref<Eigen::Matrix3f> rotation_y(Eigen::Index tet);
    Eigen::Ref<Eigen::Matrix3f> rotation(Eigen::Index tet);
    Eigen::Ref<const Eigen::Matrix3f> curl(Eigen::Index tet, int idx) const;
    Eigen::Ref<const Eigen::Matrix3f> rotation_y(Eigen::Index tet) const;
    Eigen::Ref<const Eigen::Matrix3f> rotation(Eigen::Index tet) const;
    float& edge_weight(Eigen::Index tet, int i, int j);
    float edge_weight(Eigen::Index tet, int i, int j) const;
    Eigen::Ref<const Eigen::Vector3f> p(Eigen::Index tet, int i) const;
    Eigen::Ref<const Eigen::Vector3f> q(Eigen::Index tet, int i) const;

    Eigen::Matrix3f hessian_first_part(Eigen::Index tet,
                                       int i,
                                       int j) const;
    Eigen::Matrix3f hessian_second_part(Eigen::Index tet,
                                        int i,
                                        int j) const;
    Eigen::Matrix3f hessian_second_part_split(Eigen::Index tet,
                                              int i,
                                              int j) const;
    
public:
    VolumeARAPModel(const TetraMesh& mesh, float alpha, float beta);
    
    virtual void set_query_point(const Eigen::VectorXf& x) override;
    
    virtual float energy() const override;
    
    virtual Eigen::VectorXf gradient() const override;
    virtual Eigen::SparseMatrix<float> hessian() const override;

    virtual int degrees_of_freedom() const override;
};
