#include "arap.hpp"
#include "arap_newton.hpp"
#include <imgui/imgui.h>

#include <Eigen/Dense>
#include <iostream>

Eigen::VectorXf NewtonSolver::gradient() const {
    int n = current_points_.size() / 3;

    Eigen::Matrix<float, Eigen::Dynamic, 3> rotations_inv(rotations_.rows(), 3);
    for (Eigen::Index i = 0; i < n; i++) {
	rotations_inv.block<3, 3>(3 * i, 0) = rotations_.block<3, 3>(3 * i, 0).transpose();
    }
    
    Eigen::Map<const Eigen::Matrix<float, 3, Eigen::Dynamic>> points_mat(current_points_.data(), 3, n);
    Eigen::Map<const Eigen::Matrix<float, 3, Eigen::Dynamic>> rest_points_mat(rest_points_.data(), 3, n);

    std::vector<Eigen::Triplet<float>> q_triplets;
    for (Eigen::Index j = 0; j < n; j++) {
	for (Eigen::SparseMatrix<float>::InnerIterator it(edge_weights_, j); it; ++it) {
	    assert(j == it.col());
	    Eigen::Vector3f weighted_edge = it.value() * (rest_points_mat.col(it.row()) - rest_points_mat.col(it.col()));

	    for (int a = 0; a < 3; a++) {
		q_triplets.emplace_back(
		    it.row(),
		    3 * it.row() + a,
		    weighted_edge(a));
	    }

	    for (int a = 0; a < 3; a++) {
		q_triplets.emplace_back(
		    it.col(),
		    3 * it.row() + a,
		    -weighted_edge(a));
	    }
	}
    }

    Eigen::SparseMatrix<float> q(n, 3 * n);
    q.setFromTriplets(q_triplets.begin(), q_triplets.end());
    
    Eigen::VectorXf gradient(n * 3);    
    Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> gradient_mat(gradient.data(), 3, n);
    gradient_mat = 2.0f * points_mat * laplacian_matrix_ - rotations_inv.transpose() * q.transpose();

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

Eigen::SparseMatrix<float> NewtonSolver::hessian() const {
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
	for (Eigen::SparseMatrix<float>::InnerIterator it(edge_weights_, j); it; ++it) {
	    Eigen::Matrix3f rot = rotations_.block<3, 3>(3 * it.row(), 0);
	    Eigen::Vector3f rest_dp = rest_points_mat.col(it.row()) - rest_points_mat.col(it.col());
	    Eigen::Vector3f rotated_rest_dp = rot * rest_dp;

	    for (int a = 0; a < 3; a++) {
		Eigen::Vector3f term = 2.0f * it.value() * apply_l(a, rotated_rest_dp);

		for (int b = 0; b < 3; b++) {
		    points_rot_var_triplets.emplace_back(
			3 * it.row() + a,
			3 * it.col() + b,
			term(b));
		    points_rot_var_triplets.emplace_back(
			3 * it.row() + a,
			3 * it.col() + b,
			-term(b));
		}
	    }
	}
    }
    Eigen::SparseMatrix<float> points_rot_var(3 * n, 3 * n);
    points_rot_var.setFromTriplets(points_rot_var_triplets.begin(), points_rot_var_triplets.end());

    std::vector<Eigen::Triplet<float>> rot_rot_var_inv_triplets;    
    for (int j = 0; j < n; j++) {
	Eigen::Matrix3f rot = rotations_.block<3, 3>(3 * j, 0);
	Eigen::Matrix3f block = Eigen::Matrix3f::Zero();
	    
	for (Eigen::SparseMatrix<float>::InnerIterator it(edge_weights_, j); it; ++it) {
	    Eigen::Vector3f dp = points_mat.col(it.col()) - points_mat.col(it.row());
	    Eigen::Vector3f rest_dp = rest_points_mat.col(it.col()) - rest_points_mat.col(it.row());
	    Eigen::Vector3f rotated_rest_dp = rot * rest_dp;
	    
	    for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
		    block(a, b) +=
			2.0f * it.value() * dp.dot(apply_l(a, apply_l(b, rotated_rest_dp)));
		}
	    }
	}

	Eigen::Matrix3f block_inv =
	    block.inverse();
	
	for (int a = 0; a < 3; a++) {
	    for (int b = 0; b < 3; b++) {
		rot_rot_var_inv_triplets.emplace_back(
		    3 * j + a,
		    3 * j + b,
		    block_inv(a, b));
	    }
	}
    }
    Eigen::SparseMatrix<float> rot_rot_var_inv(3 * n, 3 * n);
    rot_rot_var_inv.setFromTriplets(rot_rot_var_inv_triplets.begin(), rot_rot_var_inv_triplets.end());

    Eigen::SparseMatrix<float> second_term = points_rot_var.transpose() * rot_rot_var_inv * points_rot_var;
    // std::cout << "Second term norm : " << second_term.norm() << "\n";
    // std::cout << "rot rot var inv norm : " << rot_rot_var_inv.norm() << "\n";
    // std::cout << "points rot var norm : " << points_rot_var.norm() << "\n";
    
    // hessian -= second_term;
    hessian -= rot_rot_var_inv;

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

Eigen::VectorXf compute_empirical_gradient(const NewtonSolver& solver,
					   const Eigen::VectorXf& points,
					   const Eigen::Matrix<float, Eigen::Dynamic, 3>& rotations) {
    const float epsilon = 1e-4f;

    Eigen::VectorXf gradient = Eigen::VectorXf::Zero  (points.size());
    float f0 = solver.energy();

    for (Eigen::Index i = 0; i < points.size(); i++) {
	Eigen::VectorXf other = points;
	other(i) += epsilon;

	gradient(i) = (solver.energy() - f0) / epsilon;
    }

    return gradient;
}

void NewtonSolver::set_points(const Eigen::VectorXf& points) {
    current_points_ = points;
}

void NewtonSolver::set_points(const Mesh& mesh) {
    set_points(Eigen::Map<const Eigen::VectorXf>(mesh.V.data(), mesh.V.size()));
}

void NewtonSolver::step() {
    rotations_ =
	compute_best_rotations(current_points_, rest_points_, edge_weights_);

    float not_identity = 0.0f;
    for (int i = 0; i < rest_points_.size() / 3; i++) {
	not_identity = std::max(not_identity, (rotations_.block<3, 3>(3 * i, 0) - Eigen::Matrix3f::Identity()).norm());
    }
    // std::cout << "Not identity : " << not_identity << "\n";

    Eigen::VectorXf gradient = this->gradient();
    Eigen::SparseMatrix<float> hessian = this->hessian();

    // std::cout << "gradient : " << gradient << "\n\n";
    // std::cout << "hessian norm : " << hessian.norm() << "\n\n\n";
  
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver(hessian);

    if (solver.info() != Eigen::Success) {
	throw std::runtime_error("Could not factorize Newton hessian");
    }

    float min_lambda = solver.vectorD().minCoeff();
    if (min_lambda < -1e-5f) {
	std::cerr << "Hessian is not positive\n";
    }

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

    // std::cout << "delta :\n" << delta << "\n";
    // std::cout << "current (after) :\n" << current_points_ << "\n";

    float change = delta.cwiseAbs().maxCoeff();
    float energy = this->energy();

    ImGui::Text(("energy : " + std::to_string(energy)).c_str());
    ImGui::Text(("change : " + std::to_string(change)).c_str());
}

void NewtonSolver::apply(Mesh& mesh) const {
    mesh.V = Eigen::Map<const Points>(current_points_.data(), mesh.V.rows(), 3);
}
