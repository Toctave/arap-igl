#include "arap_util.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <algorithm> // std::sort

Eigen::SparseMatrix<float> uniform_weights(const Mesh& mesh) {
    Eigen::SparseMatrix<float> weights(mesh.V.rows(), mesh.V.rows());
    std::vector<Eigen::Triplet<float>> triplets;
    triplets.reserve(6 * mesh.F.rows());

    for (int fid = 0; fid < mesh.F.rows(); fid++) {
	Eigen::Vector3i v = mesh.F.row(fid);

	float w = .5f;
	
	triplets.emplace_back(v(0), v(1), w);
	triplets.emplace_back(v(1), v(0), w);
	    
	triplets.emplace_back(v(1), v(2), w);
	triplets.emplace_back(v(2), v(1), w);
	    
	triplets.emplace_back(v(2), v(0), w);
	triplets.emplace_back(v(0), v(2), w);
    }

    weights.setFromTriplets(triplets.begin(), triplets.end());


    return weights;
}

Eigen::SparseMatrix<float> cotangent_weights(const Mesh& mesh) {
    Eigen::SparseMatrix<float> weights(mesh.V.rows(), mesh.V.rows());
    std::vector<Eigen::Triplet<float>> triplets;
    triplets.reserve(18 * mesh.F.rows());

    for (int fid = 0; fid < mesh.F.rows(); fid++) {
	Eigen::Vector3i v = mesh.F.row(fid);

	Eigen::Matrix<float, 3, 3> edges;
	for (int j = 0; j < 3; j++) {
	    int k = (j + 1) % 3;

	    // edge vector between v(j) and v(k)
	    edges.row(j) = mesh.V.row(v(k)) - mesh.V.row(v(j));
	}

	float area_doubled = edges.row(0).cross(edges.row(1)).norm();
	float one_over_8area = 1.0f / (4 * area_doubled);

	for (int j = 0; j < 3; j++) {
	    int k = (j + 1) % 3;
	    int i = (j + 2) % 3;
	    
	    float d2 = edges.row(j).squaredNorm();

	    float contribution = one_over_8area * d2;

	    triplets.emplace_back(v(j), v(k), -contribution);
	    triplets.emplace_back(v(k), v(j), -contribution);
	    
	    triplets.emplace_back(v(i), v(j), contribution);
	    triplets.emplace_back(v(j), v(i), contribution);
	    
	    triplets.emplace_back(v(i), v(k), contribution);
	    triplets.emplace_back(v(k), v(i), contribution);
	}
    }

    weights.setFromTriplets(triplets.begin(), triplets.end());


    return weights;
}

Eigen::SparseMatrix<float> cotangent_weights_abs(const Mesh& mesh) {
    Eigen::SparseMatrix<float> weights(mesh.V.rows(), mesh.V.rows());
    std::vector<Eigen::Triplet<float>> triplets;

    triplets.reserve(mesh.F.rows() * 6);
    for (int fid = 0; fid < mesh.F.rows(); fid++) {
	Eigen::Vector3i v = mesh.F.row(fid);

	Eigen::Matrix<float, 3, 3> edges;
	for (int j = 0; j < 3; j++) {
	    int k = (j + 1) % 3;
	    
	    // edge vector between v(j) and v(k)
	    edges.row(j) = mesh.V.row(v(k)) - mesh.V.row(v(j));
	}

	for (int j = 0; j < 3; j++) {
	    int k = (j + 1) % 3;	    
	    int i = (j + 2) % 3;
	    
	    float dot = edges.row(j).dot(edges.row(k));
	    Eigen::Vector3f cross = edges.row(j).cross(edges.row(k));

	    float abs_cotan = std::abs(dot) / cross.norm();

	    triplets.emplace_back(v(i), v(j), abs_cotan);
	    triplets.emplace_back(v(j), v(i), abs_cotan);
	}
    }

    weights.setFromTriplets(triplets.begin(), triplets.end());

    return weights;
}

Eigen::SparseMatrix<float> cotangent_weights_clamped(const Mesh& mesh) {
    Eigen::SparseMatrix<float> weights = cotangent_weights(mesh);

    for (int k=0; k < weights.outerSize(); ++k) {
	for (Eigen::SparseMatrix<float>::InnerIterator it(weights,k); it; ++it) {
	    if (it.value() < 0.0f) {
		weights.coeffRef(it.row(), it.col()) = 0.0f;
	    }
	}
    }

    return weights;
}

Eigen::SparseMatrix<float> mean_value_weights(const Mesh& mesh) {
    Eigen::SparseMatrix<float> weights(mesh.V.rows(), mesh.V.rows());
    std::vector<Eigen::Triplet<float>> triplets;

    triplets.reserve(mesh.F.rows() * 12);
    for (int fid = 0; fid < mesh.F.rows(); fid++) {
	Eigen::Vector3i v = mesh.F.row(fid);

	Eigen::Matrix<float, 3, 3> edges;
	float edge_norms[3];
	for (int j = 0; j < 3; j++) {
	    int k = (j + 1) % 3;
	    
	    // edge vector between v(j) and v(k)
	    edges.row(j) = mesh.V.row(v(k)) - mesh.V.row(v(j));
	    edge_norms[j] = edges.row(j).norm();
	}

	for (int j = 0; j < 3; j++) {
	    int k = (j + 1) % 3;	    
	    int i = (j + 2) % 3;
	    
	    float cos_alpha = -edges.row(j).dot(edges.row(i)) / (edge_norms[j] * edge_norms[i]);
	    float tan_half_alpha = std::sqrt((1.0f - cos_alpha) / (1.0f + cos_alpha));

	    float wij = tan_half_alpha / edge_norms[i];

	    triplets.emplace_back(v(i), v(j), wij);
	    triplets.emplace_back(v(j), v(i), wij);

	    float wjk = tan_half_alpha / edge_norms[j];
	    
	    triplets.emplace_back(v(j), v(k), wjk);
	    triplets.emplace_back(v(k), v(j), wjk);
	}
    }

    weights.setFromTriplets(triplets.begin(), triplets.end());

    return weights;
}

Eigen::Matrix<float, Eigen::Dynamic, 3>
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

Eigen::SparseMatrix<float> compute_laplacian_matrix(const Eigen::SparseMatrix<float>& edge_weights) {
    Eigen::SparseMatrix<float> laplacian_matrix(edge_weights.rows(), edge_weights.cols());
    
    std::vector<Eigen::Triplet<float>> triplets;
    for (int i = 0; i < edge_weights.cols(); i++) {
	float colsum = 0.0f;
	for (Eigen::SparseMatrix<float>::InnerIterator it(edge_weights, i); it; ++it) {
	    colsum += it.value();
	}

	triplets.emplace_back(i, i, colsum);
    }

    laplacian_matrix.setFromTriplets(triplets.begin(), triplets.end());
    laplacian_matrix -= edge_weights;

    return laplacian_matrix;
}

