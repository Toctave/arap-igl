#include "volume_arap_model.hpp"

#include <Eigen/Geometry>
#include <Eigen/Core>

#include <iostream>

Eigen::Ref<Eigen::Matrix3f> VolumeARAPModel::curl(Eigen::Index tet, int idx) {
    return curls_.block<3, 3>(0, (4 * tet + idx) * 3);
}

Eigen::Ref<Eigen::Matrix3f> VolumeARAPModel::rotation_y(Eigen::Index tet) {
    return rotations_y_.block<3, 3>(0, 3 * tet);
}

Eigen::Ref<Eigen::Matrix3f> VolumeARAPModel::rotation(Eigen::Index tet) {
    return rotations_.block<3, 3>(0, 3 * tet);
}

Eigen::Ref<const Eigen::Matrix3f> VolumeARAPModel::curl(Eigen::Index tet, int idx) const {
    return curls_.block<3, 3>(0, (4 * tet + idx) * 3);
}

Eigen::Ref<const Eigen::Matrix3f> VolumeARAPModel::rotation_y(Eigen::Index tet) const {
    return rotations_y_.block<3, 3>(0, 3 * tet);
}

Eigen::Ref<const Eigen::Matrix3f> VolumeARAPModel::rotation(Eigen::Index tet) const {
    return rotations_.block<3, 3>(0, 3 * tet);
}

// return the index of the pair (i, j) in the list
// [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
static int edge_index(int i, int j) {
    if (i > j) {
        int tmp = i;
        i = j;
        j = tmp;
    }

    switch (i) {
    case 0:
        return j - 1;
    case 1:
        return j + 1;
    case 2:
        return j + 2;
    }

    return -1;
}

float& VolumeARAPModel::edge_weight(Eigen::Index tet, int i, int j) {
    return edge_weights_(tet, edge_index(i, j));
}

float VolumeARAPModel::edge_weight(Eigen::Index tet, int i, int j) const {
    return edge_weights_(tet, edge_index(i, j));
}

Eigen::Ref<const Eigen::Vector3f> VolumeARAPModel::p(Eigen::Index tet, int i) const {
    return mesh_.point(tet, i);
}

Eigen::Ref<const Eigen::Vector3f> VolumeARAPModel::q(Eigen::Index tet, int i) const {
    return current_points_.row(mesh_.indices(tet, i)).transpose();
}

VolumeARAPModel::VolumeARAPModel(const TetraMesh& mesh, float alpha, float beta)
    : mesh_(mesh),
      edge_weights_(mesh_.indices.rows(), 6),
      volumes_(mesh_.indices.rows()),
      volume_gradients_(3, mesh_.indices.size()),
      curls_(3, mesh_.indices.size() * 4 * 3),
      alpha2_(alpha * alpha),
      beta2_(beta * beta),
      current_points_(mesh_.points),
      rotations_(3, mesh_.indices.rows() * 3),
      rotations_y_(3, mesh_.indices.rows() * 3)
{
    for (Eigen::Index tet = 0; tet < mesh_.indices.rows(); tet++) {
        // tetrahedron volume
        volumes_(tet) =
            (p(tet, 1) - p(tet, 0)).dot((p(tet, 2) - p(tet, 0)).cross(p(tet, 3) - p(tet, 0))) / 6.0f;
        assert(volumes_(tet) > 0.0f);

        assert(-(p(tet, 2) - p(tet, 1)).dot((p(tet, 3) - p(tet, 1)).cross(p(tet, 0) - p(tet, 1))) > 0.0f);
        assert((p(tet, 3) - p(tet, 2)).dot((p(tet, 0) - p(tet, 2)).cross(p(tet, 1) - p(tet, 2))) > 0.0f);
        assert(-(p(tet, 0) - p(tet, 3)).dot((p(tet, 1) - p(tet, 3)).cross(p(tet, 2) - p(tet, 3))) > 0.0f);


        // volume gradient w.r.t. vertices
        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;
            int k = (i + 2) % 4;
            int l = (i + 3) % 4;

            Eigen::Vector3f grad = (p(tet, j) - p(tet, l)).cross(p(tet, k) - p(tet, l)) / 6.0f;

            volume_gradients_.col(tet * 4 + i) =
                (i % 2 == 0) ? -grad : grad;
        }

        for (int i = 0; i < 3; i++) {
            for (int j = i + 1; j < 4; j++) {
                edge_weight(tet, i, j) =
                    - volume_gradients_.col(tet * 4 + i)
                    .dot(volume_gradients_.col(tet * 4 + j))
                    / volumes_(tet);
            }
        }

        // todo : turn this into a tidy loop ?
        Eigen::Vector3f p01 = p(tet, 1) - p(tet, 0);
        Eigen::Vector3f p02 = p(tet, 2) - p(tet, 0);
        Eigen::Vector3f p03 = p(tet, 3) - p(tet, 0);
        Eigen::Vector3f p21 = p(tet, 1) - p(tet, 2);
        Eigen::Vector3f p13 = p(tet, 3) - p(tet, 1);
        Eigen::Vector3f p32 = p(tet, 2) - p(tet, 3);

        curl(tet, 0) =
            p01 * p32.transpose()
            + p02 * p13.transpose()
            + p03 * p21.transpose();
        curl(tet, 1) =
            p01 * p32.transpose()
            + p21 * p03.transpose()
            + p13 * p02.transpose();
        curl(tet, 2) =
            p02 * p13.transpose()
            + p21 * p03.transpose()
            + p32 * p01.transpose();
        curl(tet, 3) =
            p03 * p21.transpose()
            + p13 * p02.transpose()
            + p32 * p01.transpose();
    }
}

void VolumeARAPModel::set_query_point(const Eigen::VectorXf& x) {
    current_points_ = Eigen::Map<const Points>(x.data(), x.size() / 3, 3);
    
    for (Eigen::Index tet = 0; tet < mesh_.indices.rows(); tet++) {
        Eigen::Matrix3f m_q;
        m_q.col(0) = q(tet, 1) - q(tet, 0);
        m_q.col(1) = q(tet, 2) - q(tet, 0);
        m_q.col(2) = q(tet, 3) - q(tet, 0);

        Eigen::Matrix3f m_p;
        m_p.col(0) = p(tet, 1) - p(tet, 0);
        m_p.col(1) = p(tet, 2) - p(tet, 0);
        m_p.col(2) = p(tet, 3) - p(tet, 0);

        Eigen::Matrix3f df = m_q * m_p.inverse();
        // todo : precompute and cache m_p.inverse() for all tet

        assert((df * (p(tet, 1) - p(tet, 0)) - (q(tet, 1) - q(tet, 0))).norm() < 1.0e-3f);
        assert((df * (p(tet, 2) - p(tet, 0)) - (q(tet, 2) - q(tet, 0))).norm() < 1.0e-3f);
        assert((df * (p(tet, 3) - p(tet, 0)) - (q(tet, 3) - q(tet, 0))).norm() < 1.0e-3f);

        Eigen::JacobiSVD<Eigen::Matrix3f> svd_solver(df, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f u = svd_solver.matrixU();
        Eigen::Matrix3f v = svd_solver.matrixV();
        Eigen::Matrix3f rot = u * v.transpose();

        if (rot.determinant() < 0.0f) {
            v.row(2) *= -1.0f;
            rot = u * v.transpose();
        }

        Eigen::Matrix3f y = rot.transpose() * df;
        
        rotation(tet) = rot;
        rotation_y(tet) = y;

        assert((rot * rot.transpose() - Eigen::Matrix3f::Identity()).norm() < 1.0e-3f);
        assert((rot * y - df).norm() < 1.0e-3f);
    }
}

float VolumeARAPModel::energy() const {
    float energy = 0.0f;

    for (Eigen::Index tet = 0; tet < mesh_.indices.rows(); tet++) {
        Eigen::Matrix3f rot = rotations_.block<3, 3>(0, 3 * tet);

        for (int i = 0; i < 3; i++) {
            for (int j = i + 1; j < 4; j++) {
                Eigen::Vector3f dq = q(tet, j) - q(tet, i);
                Eigen::Vector3f dp = p(tet, j) - p(tet, i);
                float term =
                    edge_weight(tet, i, j) * (dq - rot * dp).squaredNorm();

                energy += term / 12.0f;
            }
        }
    }
    return energy;
}
    
Eigen::VectorXf VolumeARAPModel::gradient() const {
    Eigen::VectorXf gradient =
        Eigen::VectorXf::Zero(current_points_.size());

    Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>>
        gradient_mat(gradient.data(), 3, current_points_.size() / 3);
    
    for (Eigen::Index tet = 0; tet < mesh_.indices.rows(); tet++) {
        Eigen::Matrix3f y = rotations_y_.block<3, 3>(0, 3 * tet);
        Eigen::Matrix3f rot = rotations_.block<3, 3>(0, 3 * tet);
        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;
            int k = (i + 2) % 4;
            int l = (i + 3) % 4;
            
            Eigen::Vector3f laplace_q =
                edge_weight(tet, i, j) * (q(tet, i) - q(tet, j))
                + edge_weight(tet, i, k) * (q(tet, i) - q(tet, k))
                + edge_weight(tet, i, l) * (q(tet, i) - q(tet, l));

            gradient_mat.col(mesh_.indices(tet, i)) +=
                beta2_ * laplace_q
                - (alpha2_ - (alpha2_ - beta2_) * y.trace() / 3.0f)
                * rot
                * volume_gradients_.col(tet * 4 + i);
        }
    }

    return gradient;
}

Eigen::Matrix3f VolumeARAPModel::hessian_first_part(Eigen::Index tet, int i, int j) const {
    if (i == j) {
        return Eigen::Matrix3f::Identity()
            * (edge_weight(tet, i, (i+1) % 4)
               + edge_weight(tet, i, (i+2) % 4)
               + edge_weight(tet, i, (i+3) % 4));
    } else {
        return -Eigen::Matrix3f::Identity()
            * edge_weight(tet, i, j);
    }
}

Eigen::Matrix3f VolumeARAPModel::hessian_second_part(Eigen::Index tet, int i, int j) const {
    Eigen::Matrix3f rotated_curl_i =
        rotation(tet).transpose() * curl(tet, i);
    Eigen::Matrix3f rotated_curl_j =
        rotation(tet).transpose() * curl(tet, j);

    // todo : precompute w
    Eigen::Matrix3f w = (rotation_y(tet).trace() * Eigen::Matrix3f::Identity()
                         - rotation_y(tet)).inverse();

    return rotated_curl_i * w * rotated_curl_j.transpose() / (36.0f * volumes_(tet));
}

Eigen::Matrix3f VolumeARAPModel::hessian_second_part_split(Eigen::Index tet, int i, int j) const {
    return (rotation(tet) * volume_gradients_.col(tet * 4 + i)
            * (rotation(tet) * volume_gradients_.col(tet * 4 + j)).transpose())
        / volumes_(tet);
}

Eigen::SparseMatrix<float> VolumeARAPModel::hessian() const {
    std::vector<Eigen::Triplet<float>> hessian_triplets;

    for (Eigen::Index tet = 0; tet < mesh_.indices.rows(); tet++) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                Eigen::Matrix3f hessian_ij =
                    beta2_ * hessian_first_part(tet, i, j)
                    - (alpha2_ - (alpha2_ - beta2_) * rotation_y(tet).trace() / 3.0f)
                      * hessian_second_part(tet, i, j)
                    + (alpha2_ - beta2_)
                      * hessian_second_part_split(tet, i, j);
                
                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                        hessian_triplets
                            .emplace_back(mesh_.indices(tet, i) * 3 + a,
                                          mesh_.indices(tet, j) * 3 + b,
                                          hessian_ij(a, b));
                        
                    }
                }
            }
        }
    }

    Eigen::SparseMatrix<float> hessian(degrees_of_freedom(), degrees_of_freedom());
    hessian.setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
    
    return hessian;
}

int VolumeARAPModel::degrees_of_freedom() const {
    return mesh_.points.size();
}
