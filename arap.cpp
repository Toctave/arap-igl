#include "arap.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <algorithm> // std::sort

static float compute_area(const Mesh& mesh) {
    float area = 0.0f;
    for (int fid = 0; fid < mesh.F.rows(); fid++) {
	Eigen::Vector3f e1 = mesh.V.row(mesh.F(fid, 1)) - mesh.V.row(mesh.F(fid, 0));
	Eigen::Vector3f e2 = mesh.V.row(mesh.F(fid, 2)) - mesh.V.row(mesh.F(fid, 0));

	area += e1.cross(e2).norm();
    }
    return area / 2.0f;
}

static Eigen::SparseMatrix<float> cotangent_weights(const Mesh& mesh) {
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

// static Eigen::SparseMatrix<float> cotangent_weights_abs(const Mesh& mesh) {
//     Eigen::SparseMatrix<float> weights(mesh.V.rows(), mesh.V.rows());
//     std::vector<Eigen::Triplet<float>> triplets;

//     triplets.reserve(mesh.F.rows() * 6);
//     for (int fid = 0; fid < mesh.F.rows(); fid++) {
// 	Eigen::Vector3i v = mesh.F.row(fid);

// 	Eigen::Matrix<float, 3, 3> edges;
// 	for (int j = 0; j < 3; j++) {
// 	    int k = (j + 1) % 3;
	    
// 	    // edge vector between v(j) and v(k)
// 	    edges.row(j) = mesh.V.row(v(k)) - mesh.V.row(v(j));
// 	}

// 	for (int j = 0; j < 3; j++) {
// 	    int k = (j + 1) % 3;	    
// 	    int i = (j + 2) % 3;
	    
// 	    float dot = edges.row(j).dot(edges.row(k));
// 	    Eigen::Vector3f cross = edges.row(j).cross(edges.row(k));

// 	    float abs_cotan = std::abs(dot) / cross.norm();

// 	    triplets.emplace_back(v(i), v(j), abs_cotan);
// 	    triplets.emplace_back(v(j), v(i), abs_cotan);
// 	}
//     }

//     weights.setFromTriplets(triplets.begin(), triplets.end());

//     return weights;
// }

static Eigen::SparseMatrix<float> cotangent_weights_clamped(const Mesh& mesh) {
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

static Eigen::SparseMatrix<float> mean_value_weights(const Mesh& mesh) {
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

static Eigen::SparseMatrix<float> laplacian_matrix(
    const Eigen::SparseMatrix<float>& weights) {
    
    Eigen::SparseMatrix<float> mat = -weights;

    for (int k=0; k < mat.outerSize(); ++k) {
	float colsum = 0.0f;
	for (Eigen::SparseMatrix<float>::InnerIterator it(mat,k); it; ++it) {
	    colsum += it.value();
	}
	mat.coeffRef(k, k) = -colsum;
    }

    return mat;
}

static Eigen::Matrix3f compute_best_rotation(const LaplacianSystem& system, int r) {
    Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();

    for (Eigen::SparseMatrix<float>::InnerIterator it(system.edge_weights, r); it; ++it) {
	Eigen::Index v_idx[2] = {
	    it.col(),
	    it.row()
	};

	Eigen::Vector3f e = system.mesh->V.row(v_idx[0]) - system.mesh->V.row(v_idx[1]);
	Eigen::Vector3f e0 = system.V0.row(v_idx[0]) - system.V0.row(v_idx[1]);

	cov += it.value() * (e0 * e.transpose() +
			     system.rotation_variation_penalty * system.optimal_rotations[it.row()].transpose());
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

    return rot;
}

void system_init(LaplacianSystem& system, Mesh* mesh, float alpha) {
    system.mesh = mesh;
    system.iterations = 0;

    system.rotation_variation_penalty = alpha * compute_area(*mesh);

    system.optimal_rotations.reserve(mesh->V.rows());
    for (size_t i = 0; i < mesh->V.rows(); i++) {
	system.optimal_rotations.push_back(Eigen::Matrix3f::Identity());
    }
}

void print_sparse(const Eigen::SparseMatrix<float>& mat) {
    for (int k=0; k<mat.outerSize(); ++k) {
	for (Eigen::SparseMatrix<float>::InnerIterator it(mat, k); it; ++it) {
	    if (it.value() != 0.0f) {
		std::cout << it.row() << " " << it.col() << " " << it.value() << "\n";
	    }
	}
    }
}

bool system_bind(LaplacianSystem& system,
		 const std::vector<FixedVertex>& fixed_vertices,
		 WeightType weights_type) {
    system.V0 = system.mesh->V;

    for (const auto& fv : fixed_vertices) {
	system.fixed_vertex_indices.push_back(fv.index);
    }

    switch (weights_type) {
    case COTANGENT:
    	system.edge_weights = cotangent_weights(*system.mesh);
	break;
    case COTANGENT_CLAMPED:
    	system.edge_weights = cotangent_weights_clamped(*system.mesh);
	break;
    case COTANGENT_ABS:
    	// system.edge_weights = cotangent_weights_abs(*system.mesh);
	break;
    case MEAN_VALUE:
	system.edge_weights = mean_value_weights(*system.mesh);
	break;
    default:
	throw std::runtime_error("Unknown weight type");
    };

    system.laplacian_matrix = laplacian_matrix(system.edge_weights);

    for (Eigen::Index fixed : system.fixed_vertex_indices) {
	for (Eigen::SparseMatrix<float>::InnerIterator it(system.laplacian_matrix, fixed); it; ++it) {
	    system.laplacian_matrix.coeffRef(it.row(), it.col()) =
		(it.col() == it.row()) ? 1.0f : 0.0f;
	    system.laplacian_matrix.coeffRef(it.col(), it.row()) =
		(it.col() == it.row()) ? 1.0f : 0.0f;
	}
    }

    system.solver.compute(system.laplacian_matrix);
    if (system.solver.info() != Eigen::Success) {
	std::cerr << "error when factorizing laplacian matrix \n";
	return false;
    }

    system.rhs.resize(system.mesh->V.rows(), 3);

    return true;
}

bool system_iterate(LaplacianSystem& system) {
    /* --- Compute approximate rotations --- */

    for (int i = 0; i < system.mesh->V.rows(); i++) {
	system.optimal_rotations[i] = compute_best_rotation(system, i);
    }

    /* --- Fill system's right hand side --- */
    
    system.rhs.setZero();

    for (int v = 0; v < system.rhs.rows(); v++) {
	for (Eigen::SparseMatrix<float>::InnerIterator
		 it(system.edge_weights, v);
	     it;
	     ++it) {
	    
	    Eigen::RowVector3f d = .5f * it.value() *
		(system.V0.row(it.col()) - system.V0.row(it.row())) *
		(system.optimal_rotations[it.row()] + system.optimal_rotations[it.col()]).transpose();

	    system.rhs.row(v) += d;
	}
    }

    /* --- Special case for fixed vertices --- */
    
    for (Eigen::Index fixed : system.fixed_vertex_indices) {
    	for (Eigen::SparseMatrix<float>::InnerIterator it(system.edge_weights, fixed); it; ++it) {
    	    system.rhs.row(it.row()) +=
    		it.value() * system.mesh->V.row(fixed);
    	}
    }

    for (Eigen::Index fixed : system.fixed_vertex_indices) {
	system.rhs.row(fixed) = system.mesh->V.row(fixed);
    }

    /* --- Solve the system --- */
    
    for (int i = 0; i < 3; i++) {
	system.mesh->V.col(i) = system.solver.solve(system.rhs.col(i));
	
	if (system.solver.info() != Eigen::Success) {
	    std::cout << "Error in solver\n";
	    return false;
	}
    }

    assert((system.laplacian_matrix * system.mesh->V
	    - system.rhs).norm() < 1.0e-3f);
    
    system.iterations++;

    return true;
}

void system_solve(LaplacianSystem& system, int iterations) {
    for (int i = 0; i < iterations; i++) {
	system_iterate(system);
    }
}
