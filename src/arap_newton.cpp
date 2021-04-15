#include "arap.hpp"
#include "arap_newton.hpp"

#include <Eigen/Dense>
#include <iostream>

Eigen::VectorXf compute_gradient(const Eigen::VectorXf& points,
				 const Eigen::VectorXf& rest_points,
				 const Eigen::SparseMatrix<float>& edge_weights,
				 const Eigen::SparseMatrix<float>& laplacian_matrix,
				 const Eigen::Matrix<float, Eigen::Dynamic, 3>& rotations) {
    int n = points.size() / 3;

    Eigen::Matrix<float, Eigen::Dynamic, 3> rotations_inv(rotations.rows(), 3);
    for (Eigen::Index i = 0; i < n; i++) {
	rotations_inv.block<3, 3>(3 * i, 0) = rotations.block<3, 3>(3 * i, 0).transpose();
    }
    
    Eigen::Map<const Eigen::Matrix<float, 3, Eigen::Dynamic>> points_mat(points.data(), 3, n);
    Eigen::Map<const Eigen::Matrix<float, 3, Eigen::Dynamic>> rest_points_mat(rest_points.data(), 3, n);

    Eigen::VectorXf gradient(n * 3);    
    Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> gradient_mat(gradient.data(), 3, n);

    Eigen::SparseMatrix<float> q(n, 3 * n);

    // todo : do this with triplets
    for (Eigen::Index i = 0; i < n; i++) {
	for (Eigen::SparseMatrix<float>::InnerIterator it(edge_weights, i); it; ++it) {
	    for (int b = 0; b < 3; b++) {
		q.coeffRef(it.col(), 3 * it.row() + b) =
		    -it.value() * (rest_points_mat(b, it.row()) - rest_points_mat(b, it.col()));
	    }
	}
    }

    gradient_mat = 2.0f * (points_mat * laplacian_matrix - rotations.transpose() * q.transpose());

    return gradient;
}

Eigen::Vector3f apply_l(int a, const Eigen::Vector3f& v) {
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

Eigen::SparseMatrix<float> compute_hessian(const Eigen::VectorXf& points,
					   const Eigen::VectorXf& rest_points,
					   const Eigen::SparseMatrix<float>& edge_weights,
					   const Eigen::SparseMatrix<float>& laplacian_matrix,
					   const Eigen::Matrix<float, Eigen::Dynamic, 3>& rotations) {
    int n = points.size() / 3;
    
    Eigen::Map<const Eigen::Matrix<float, 3, Eigen::Dynamic>> points_mat(points.data(), 3, n);
    Eigen::Map<const Eigen::Matrix<float, 3, Eigen::Dynamic>> rest_points_mat(rest_points.data(), 3, n);
    
    Eigen::SparseMatrix<float> hessian(3 * n, 3 * n);
    for (int i = 0; i < n; i++) {
	float colsum = 0.0f;
	for (Eigen::SparseMatrix<float>::InnerIterator it(laplacian_matrix, i); it; ++it) {
	    float w = it.value();
	    for (int b = 0; b < 3; b++) {
		hessian.coeffRef(3 * it.row() + b, 3 * it.col() + b) = w;
	    }
	}
    }

    Eigen::SparseMatrix<float> points_rot_var(3 * n, 3 * n);
    for (int i = 0; i < n; i++) {
	for (Eigen::SparseMatrix<float>::InnerIterator it(edge_weights, i); it; ++it) {
	    Eigen::Matrix3f rot = rotations.block<3, 3>(3 * it.row(), 0);
	    Eigen::Vector3f rest_dp = rest_points_mat.col(it.row()) - rest_points_mat.col(it.col());
	    Eigen::Vector3f rotated_rest_dp = rot * rest_dp;
	    
	    for (int a = 0; a < 3; a++) {
		Eigen::Vector3f block_row = 
		    2.0f * it.value() * apply_l(a, rotated_rest_dp);

		for (int b = 0; b < 3; b++) {
		    points_rot_var.coeffRef(3 * it.row() + a, 3 * it.col() + b) =
			block_row(b);
		}
	    }
	}
    }
    
    
    Eigen::SparseMatrix<float> rot_rot_var_inv(3 * n, 3 * n);
    for (int i = 0; i < n; i++) {
	Eigen::Matrix3f rot = rotations.block<3, 3>(3 * i, 0);
	Eigen::Matrix3f block = Eigen::Matrix3f::Zero();
	    
	for (Eigen::SparseMatrix<float>::InnerIterator it(edge_weights, i); it; ++it) {
	    Eigen::Vector3f dp = points_mat.col(it.row()) - points_mat.col(it.col());
	    Eigen::Vector3f rest_dp = rest_points_mat.col(it.row()) - rest_points_mat.col(it.col());
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
		rot_rot_var_inv.coeffRef(3 * i + a, 3 * i + b) =
		    block_inv(a, b);
	    }
	}
    }

    hessian -= points_rot_var.transpose() * rot_rot_var_inv * points_rot_var;

    return hessian;
}

NewtonSolver::NewtonSolver(const Mesh& mesh) {
    rest_points = Eigen::Map<const Eigen::VectorXf>(mesh.V.data(), mesh.V.size());
    edge_weights = mean_value_weights(mesh);

    laplacian_matrix = -edge_weights;

    for (int i = 0; i < edge_weights.cols(); i++) {
	float colsum = 0.0f;
	for (Eigen::SparseMatrix<float>::InnerIterator it(edge_weights, i); it; ++it) {
	    colsum += it.value();
	}

	laplacian_matrix.coeffRef(i, i) = colsum;
    }
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
	    Eigen::Index v_idx[2] = {
		it.col(),
		it.row()
	    };

	    Eigen::Vector3f e = points_mat.col(v_idx[0]) - points_mat.col(v_idx[1]);
	    Eigen::Vector3f e0 = rest_points_mat.col(v_idx[0]) - rest_points_mat.col(v_idx[1]);

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

void NewtonSolver::step(Mesh& mesh) {
    Eigen::Map<Eigen::VectorXf> points_flat(mesh.V.data(), mesh.V.size());

    Eigen::Matrix<float, Eigen::Dynamic, 3> rotations =    
	compute_best_rotations(points_flat, rest_points, edge_weights);

    Eigen::VectorXf gradient = compute_gradient(points_flat, rest_points, edge_weights, laplacian_matrix, rotations);

    Eigen::SparseMatrix<float> hessian = compute_hessian(points_flat, rest_points, edge_weights, laplacian_matrix, rotations);

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver(hessian);

    if (solver.info() != Eigen::Success) {
	throw std::runtime_error("Could not factorize Newton hessian");
    }

    Eigen::VectorXf delta = solver.solve(gradient);
    
    if (solver.info() != Eigen::Success) {
	throw std::runtime_error("Could not invert Newton hessian");
    }

    points_flat -= delta;
}
