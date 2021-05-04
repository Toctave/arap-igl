#include "arap_model.hpp"

#include "arap_util.hpp"

#include <iostream>
#include <Eigen/Geometry>
#include <imgui/imgui.h>

int ARAPModel::degrees_of_freedom() const {
    return rest_points_.size();
}

ARAPModel::ARAPModel(const Mesh& mesh) :
    rest_points_(Eigen::Map<const Eigen::VectorXf>(mesh.V.data(), mesh.V.size())),
    current_points_(rest_points_),
    edge_weights_(mean_value_weights(mesh)),
    laplacian_matrix_(compute_laplacian_matrix(edge_weights_))
{
}

void ARAPModel::set_query_point(const Eigen::VectorXf& x) {
    current_points_ = x;
    rotations_ = compute_best_rotations(current_points_, rest_points_, edge_weights_);
}
    
float ARAPModel::energy() const {
    float e = 0.0f;
    int n = current_points_.size() / 3;
    
    for (int i = 0; i < n; i++) {
        for (Eigen::SparseMatrix<float>::InnerIterator it(edge_weights_, i); it; ++it) {
            Eigen::Vector3f edge0 =
                rest_points_.block<3, 1>(3 * it.row(), 0)
                - rest_points_.block<3, 1>(3 * it.col(), 0);
	    
            Eigen::Vector3f edge =
                current_points_.block<3, 1>(3 * it.row(), 0)
                - current_points_.block<3, 1>(3 * it.col(), 0);
	    	    
            Eigen::Matrix3f rot = rotations_.block<3, 3>(3 * i, 0);
            e += it.value() * (edge - rot * edge0).squaredNorm();
        }
    }

    return e;
}

Eigen::VectorXf ARAPModel::gradient() const {
    int n = current_points_.size() / 3;
    Eigen::Map<const Eigen::Matrix<float, 3, Eigen::Dynamic>> points_mat(current_points_.data(), 3, n);
    Eigen::Map<const Eigen::Matrix<float, 3, Eigen::Dynamic>> rest_points_mat(rest_points_.data(), 3, n);

    Eigen::VectorXf gradient = Eigen::VectorXf::Zero(n * 3);    
    Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> gradient_mat(gradient.data(), 3, n);
    
    for (Eigen::Index j = 0; j < n; j++) {
    	Eigen::Matrix3f rot_j = rotations_.block<3, 3>(3 * j, 0);
    	for (Eigen::SparseMatrix<float>::InnerIterator it(edge_weights_, j); it; ++it) {
    	    Eigen::Matrix3f rot_i = rotations_.block<3, 3>(3 * it.row(), 0);
    	    Eigen::Vector3f dp = points_mat.col(it.row()) - points_mat.col(it.col());
    	    Eigen::Vector3f rest_dp = rest_points_mat.col(it.row()) - rest_points_mat.col(it.col());
	    
    	    gradient_mat.col(it.row()) += 2.0f * it.value() * (2.0f * dp - (rot_i + rot_j) * rest_dp);
    	}
    }

    return gradient;
}

static Eigen::Vector3f apply_l(int a, const Eigen::Vector3f& v) {
    switch (a) {
    case 0:
        return Eigen::Vector3f(0.0f, -v(2), v(1));
    case 1:
        return Eigen::Vector3f(v(2), 0.0f, -v(0));
    case 2:
        return Eigen::Vector3f(-v(1), v(0), 0.0f);
    default:
        throw std::runtime_error("Invalid argument in call to apply_l");
    }
}

Eigen::VectorXf ARAPModel::empirical_gradient() const {
    ARAPModel model = *this;
    Eigen::VectorXf points = current_points_;
    
    const float epsilon = 1e-3f;

    Eigen::VectorXf gradient = Eigen::VectorXf::Zero(current_points_.size());
    float f0 = model.energy();

    for (Eigen::Index i = 0; i < current_points_.size(); i++) {
        points(i) += epsilon;
        model.set_query_point(points);
	
        gradient(i) = (model.energy() - f0) / epsilon;

        points(i) -= epsilon;
    }

    return gradient;
}

Eigen::SparseMatrix<float> ARAPModel::empirical_hessian() const {
    ARAPModel model = *this;
    Eigen::VectorXf points = current_points_;
    
    double epsilon = 1.0e-3f;
    
    std::vector<Eigen::Triplet<float>> hessian_triplets;

    for (int j = 0; j < edge_weights_.rows(); j++) {
        for (Eigen::SparseMatrix<float>::InnerIterator it(laplacian_matrix_, j); it; ++it) {
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    int ii = 3 * it.row() + a;
                    int jj = 3 * it.col() + b;

                    model.set_query_point(points);
                    double f00 = model.energy();
		    
                    points(ii) += epsilon;

                    model.set_query_point(points);
                    double f10 = model.energy();

                    points(jj) += epsilon;
		    
                    model.set_query_point(points);
                    double f11 = model.energy();
		    
                    points(ii) -= epsilon;

                    model.set_query_point(points);
                    double f01 = model.energy();
		    
                    points(jj) -= epsilon;

                    double estimate = ((f00 - f01) + (f11 - f10)) / (epsilon * epsilon);

                    hessian_triplets.emplace_back(ii, jj, static_cast<float>(estimate));
                }
            }
        }
    }

    Eigen::SparseMatrix<float> hessian(degrees_of_freedom(), degrees_of_freedom());

    hessian.setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());

    return hessian;
}


Eigen::SparseMatrix<float> ARAPModel::hessian() const {
    int n = current_points_.size() / 3;
    
    Eigen::Map<const Eigen::Matrix<float, 3, Eigen::Dynamic>> points_mat(current_points_.data(), 3, n);
    Eigen::Map<const Eigen::Matrix<float, 3, Eigen::Dynamic>> rest_points_mat(rest_points_.data(), 3, n);

    std::vector<Eigen::Triplet<float>> hessian_triplets;
    for (int j = 0; j < n; j++) {
        for (Eigen::SparseMatrix<float>::InnerIterator it(laplacian_matrix_, j); it; ++it) {
            float w = it.value();
            for (int b = 0; b < 3; b++) {
                hessian_triplets.emplace_back(3 * it.row() + b, 3 * it.col() + b, w);
            }
        }
    }
    Eigen::SparseMatrix<float> hessian(3 * n, 3 * n);
    hessian.setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
    
    std::vector<Eigen::Triplet<float>> points_rot_var_triplets;
    for (int j = 0; j < n; j++) {
        Eigen::Matrix3f rot_j = rotations_.block<3, 3>(3 * j, 0);
        for (Eigen::SparseMatrix<float>::InnerIterator it(edge_weights_, j); it; ++it) {
            Eigen::Matrix3f rot_i = rotations_.block<3, 3>(3 * it.row(), 0);
            Eigen::Vector3f dp = rest_points_mat.col(it.row()) - rest_points_mat.col(it.col());

            for (int b = 0; b < 3; b++) {
                Eigen::Vector3f non_diagonal_term = -2.0f * it.value() * apply_l(b, rot_j * dp);
                Eigen::Vector3f diagonal_term = -2.0f * it.value() * apply_l(b, rot_i * dp);

                for (int a = 0; a < 3; a++) {
                    int ii = 3 * it.row() + a;
                    int jj = 3 * it.col() + b;
                    points_rot_var_triplets.emplace_back(ii,
                                                         jj,
                                                         non_diagonal_term(a));
                    points_rot_var_triplets.emplace_back(ii,
                                                         jj,
                                                         diagonal_term(a));
                }
            }
        }
    }
    Eigen::SparseMatrix<float> points_rot_var(3 * n, 3 * n);
    points_rot_var.setFromTriplets(points_rot_var_triplets.begin(), points_rot_var_triplets.end());

    std::vector<Eigen::Triplet<float>> rot_rot_var_inv_triplets;    
    for (int i = 0; i < n; i++) {
        Eigen::Matrix3f block = Eigen::Matrix3f::Zero();
        for (Eigen::SparseMatrix<float>::InnerIterator it(edge_weights_, i); it; ++it) {
            int j = it.row();
            Eigen::Matrix3f rot_i = rotations_.block<3, 3>(i * 3, 0);

            Eigen::Vector3f dp = points_mat.col(i) - points_mat.col(j);
            Eigen::Vector3f rest_dp = rest_points_mat.col(i) - rest_points_mat.col(j);
            Eigen::Vector3f rotated_rest_dp = rot_i * rest_dp;

            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    block(a, b) +=
                        2.0f * it.value() * dp.dot(apply_l(a, apply_l(b, rotated_rest_dp)));
                }
            }
        }

        Eigen::Matrix3f block_inverse = block.inverse();
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                int ii = 3 * i + a;
                int jj = 3 * i + b;

                rot_rot_var_inv_triplets.emplace_back(ii, jj, block_inverse(a, b));
            }
        }
    }
    Eigen::SparseMatrix<float> rot_rot_var_inv(3 * n, 3 * n);
    rot_rot_var_inv.setFromTriplets(rot_rot_var_inv_triplets.begin(), rot_rot_var_inv_triplets.end());

    // hessian -= points_rot_var * (rot_rot_var_inv * points_rot_var.transpose());
    hessian -= rot_rot_var_inv;
    
    Eigen::SparseMatrix<float> hessian_t = hessian.transpose();
    ImGui::Text("hessian non-symmetry : %f", (hessian - hessian_t).norm());
    
    return hessian;
}
