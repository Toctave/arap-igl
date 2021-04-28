#include "arap.hpp"
#include "arap_newton.hpp"
#include <imgui/imgui.h>

#include <Eigen/Dense>
#include <iostream>

Eigen::VectorXf NewtonSolver::empirical_gradient() {
    const float epsilon = 1e-3f;

    Eigen::VectorXf gradient = Eigen::VectorXf::Zero(current_points_.size());
    float f0 = energy();

    for (Eigen::Index i = 0; i < current_points_.size(); i++) {
	current_points_(i) += epsilon;
	
	gradient(i) = (energy() - f0) / epsilon;
	
	current_points_(i) -= epsilon;
    }

    return gradient;
}

Eigen::VectorXf NewtonSolver::gradient() const {
    cache_rotations();

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

Eigen::SparseMatrix<float> NewtonSolver::empirical_hessian() {
    float epsilon = 1.0e-3f;
    
    std::vector<Eigen::Triplet<float>> hessian_triplets;
    
    for (int j = 0; j < edge_weights_.rows(); j++) {
	for (Eigen::SparseMatrix<float>::InnerIterator it(laplacian_matrix_, j); it; ++it) {
	    for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
		    int ii = 3 * it.row() + a;
		    int jj = 3 * it.col() + b;
		    
		    float f00 = energy();
		    
		    current_points_(ii) += epsilon;

		    float f10 = energy();

		    current_points_(jj) += epsilon;
		    
		    float f11 = energy();
		    
		    current_points_(ii) -= epsilon;

		    float f01 = energy();
		    
		    current_points_(jj) -= epsilon;

		    float estimate = ((f00 - f01) + (f11 - f10)) / (epsilon * epsilon);

		    hessian_triplets.emplace_back(ii, jj, estimate);
		}
	    }
	}
    }

    Eigen::SparseMatrix<float> hessian(current_points_.size(), current_points_.size());
    hessian.setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());

    return hessian;
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

Eigen::SparseMatrix<float> NewtonSolver::hessian() const {
    cache_rotations();
    
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

	// for (int b = 0; b < 3; b++) {
	// hessian_triplets.emplace_back(3 * j + b, 3 * j + b, 1e-6f);
	// }
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
		    points_rot_var_triplets.emplace_back(
			ii,
			jj,
			non_diagonal_term(a));
		    points_rot_var_triplets.emplace_back(
			ii,
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

    // std::cout << "Second term norm : " << second_term.norm() << "\n";
    // std::cout << "rot rot var inv norm : " << rot_rot_var_inv.norm() << "\n";
    // std::cout << "points rot var norm : " << points_rot_var.norm() << "\n";
    
    hessian -= rot_rot_var_inv;
    
    Eigen::SparseMatrix<float> hessian_t = hessian.transpose();
    ImGui::Text("hessian non-symmetry : %f", (hessian - hessian_t).norm());
    
    return hessian;
}

Eigen::SparseMatrix<float> compute_laplacian_matrix(const Eigen::SparseMatrix<float>& edge_weights) {
    Eigen::SparseMatrix<float> laplacian_matrix(edge_weights.rows(), edge_weights.cols());
    
    std::vector<Eigen::Triplet<float>> triplets;
    for (int i = 0; i < edge_weights.cols(); i++) {
	float colsum = 0.0f;
	for (Eigen::SparseMatrix<float>::InnerIterator it(edge_weights, i); it; ++it) {
	    colsum += it.value();
	}

	// std::cout << "colsum : " << colsum << "\n";
	triplets.emplace_back(i, i, colsum);
    }

    laplacian_matrix.setFromTriplets(triplets.begin(), triplets.end());
    laplacian_matrix -= edge_weights;

    return laplacian_matrix;
}

NewtonSolver::NewtonSolver(const Mesh& mesh) :
    rest_points_(Eigen::Map<const Eigen::VectorXf>(mesh.V.data(), mesh.V.size())),
    current_points_(rest_points_),
    edge_weights_(mean_value_weights(mesh)),
    laplacian_matrix_(compute_laplacian_matrix(edge_weights_)),
    rotations_cached_(false),
    step_size_(0.5f)
{
}

static Eigen::Matrix<float, Eigen::Dynamic, 3>
compute_best_rotations(const Eigen::VectorXf& points,
		       const Eigen::VectorXf& rest_points,
		       const Eigen::SparseMatrix<float>& edge_weights) {
    int n = points.size() / 3;
    Eigen::Matrix<float, Eigen::Dynamic, 3> rotations(3 * n, 3);

    Eigen::Map<const Eigen::Matrix<float, 3, Eigen::Dynamic>> points_mat(points.data(), 3, n);
    Eigen::Map<const Eigen::Matrix<float, 3, Eigen::Dynamic>> rest_points_mat(rest_points.data(), 3, n);

    for (int i = 0; i < n; i++) {
	Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();

	for (Eigen::SparseMatrix<float>::InnerIterator it(edge_weights, i); it; ++it) {
	    Eigen::Vector3f e = points_mat.col(it.row()) - points_mat.col(it.col());
	    Eigen::Vector3f e0 = rest_points_mat.col(it.row()) - rest_points_mat.col(it.col());

	    cov += it.value() * e0 * e.transpose();
	}

	Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);

	Eigen::Matrix3f um = svd.matrixU();
	Eigen::Matrix3f vm = svd.matrixV();
	Eigen::Matrix3f rot = vm * um.transpose();

	if (rot.determinant() < 0) {
	    um.col(2) *= -1;
	    rot = vm * um.transpose();
	}

	assert(fabs(rot.determinant() - 1.0f) < 1e-3);

	rotations.block<3, 3>(3 * i, 0) = rot;
    }

    return rotations;
}

float NewtonSolver::energy() const {
    cache_rotations();
    
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

void NewtonSolver::set_points(const Eigen::VectorXf& points) {
    current_points_ = points;
    rotations_cached_ = false;
}

void NewtonSolver::set_points(const Mesh& mesh) {
    set_points(Eigen::Map<const Eigen::VectorXf>(mesh.V.data(), mesh.V.size()));
}

void NewtonSolver::cache_rotations() const {
    if (!rotations_cached_) {
	rotations_ =
	    compute_best_rotations(current_points_, rest_points_, edge_weights_);
	rotations_cached_ = true;
    }
}

void NewtonSolver::step() {
    Eigen::VectorXf gradient = this->gradient();
    Eigen::SparseMatrix<float> hessian = this->hessian();

    float not_identity = 0.0f;
    for (int i = 0; i < rest_points_.size() / 3; i++) {
	not_identity = std::max(not_identity, (rotations_.block<3, 3>(3 * i, 0) - Eigen::Matrix3f::Identity()).norm());
    }
    // std::cout << "Not identity : " << not_identity << "\n";

    // std::cout << "gradient : " << gradient << "\n\n";
    // std::cout << "hessian norm : " << hessian.norm() << "\n\n\n";
  
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver(hessian);

    if (solver.info() != Eigen::Success) {
	throw std::runtime_error("Could not factorize Newton hessian");
    }

    float min_lambda = solver.vectorD().minCoeff();
    ImGui::Text("Hessian min eigenvalue : %f", min_lambda);

    // std::cout << "min lambda : " << min_lambda << "\n";

    Eigen::VectorXf delta =
	// .001f * gradient;
	solver.solve(gradient);

    if (solver.info() != Eigen::Success) {
	throw std::runtime_error("Could not invert Newton hessian");
    }

    // std::cout << "current (before) :\n" << current_points_ << "\n";
    
    ImGui::SliderFloat("Step size", &step_size_, 0.0f, 3.0f);
    current_points_ -= step_size_ * delta;
    rotations_cached_ = false;
}

template<typename T>
Eigen::VectorXf newton_step(const T& model, const Eigen::VectorXf& current_points) {
    Eigen::VectorXf gradient = model.gradient(current_points);
    Eigen::SparseMatrix<float> hessian = model.hessian(current_points);

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver(hessian);

    if (solver.info() != Eigen::Success) {
	throw std::runtime_error("Could not factorize Newton hessian");
    }
    
    Eigen::VectorXf delta =
	solver.solve(gradient);

    if (solver.info() != Eigen::Success) {
	throw std::runtime_error("Could not invert Newton hessian");
    }
    
    return current_points + delta;
}

void NewtonSolver::apply(Mesh& mesh) const {
    mesh.V = Eigen::Map<const Points>(current_points_.data(), mesh.V.rows(), 3);
}

int NewtonSolver::ndof() const {
    return current_points_.size();
}

const Eigen::VectorXf& NewtonSolver::current_points() const {
    return current_points_;
}
