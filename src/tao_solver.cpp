#include <iostream>

#include "tao_solver.hpp"

#include <imgui/imgui.h>

static void eigen_to_tao(float* tao, float eigen) {
    *tao = eigen;
}

static void eigen_to_tao(Vec tao, const Eigen::VectorXf& eigen) {
    std::vector<int> indices(eigen.size());
    for (int i = 0; i < eigen.size(); i++) {
	indices[i] = i;
    }
    
    VecSetValues(tao, indices.size(), indices.data(), eigen.data(), INSERT_VALUES);
    VecAssemblyBegin(tao);
    VecAssemblyEnd(tao);
}

static void eigen_to_tao(Mat tao, const Eigen::SparseMatrix<float>& eigen) {
    MatSetUp(tao);
    MatSetOption(tao, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    
    for (int j = 0; j < eigen.cols(); j++) {
        for (Eigen::SparseMatrix<float>::InnerIterator it(eigen, j); it; ++it) {
            MatSetValue(tao, it.row(), it.col(), it.value(), INSERT_VALUES);
        }
    }

    MatAssemblyBegin(tao, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(tao, MAT_FINAL_ASSEMBLY);
}

static Eigen::VectorXf tao_to_eigen(Vec tao) {
    int size;
    VecGetSize(tao, &size);
    
    const float* data;
    VecGetArrayRead(tao, &data);

    Eigen::VectorXf rval = Eigen::Map<const Eigen::VectorXf>(data, size);

    return rval;
}

struct TaoCallbackData {
    EnergyModel& model;
};

static PetscErrorCode tao_energy_gradient_callback(Tao tao, Vec x, PetscReal* energy, Vec gradient, void* user) {
    TaoCallbackData* cb_data = reinterpret_cast<TaoCallbackData*>(user);

    cb_data->model.set_query_point(tao_to_eigen(x));

    eigen_to_tao(energy, cb_data->model.energy());
    eigen_to_tao(gradient, cb_data->model.gradient());

    return 0;
}

static PetscErrorCode tao_hessian_callback(Tao tao, Vec x, Mat hessian, Mat preconditioner, void* user) {
    TaoCallbackData* cb_data = reinterpret_cast<TaoCallbackData*>(user);

    cb_data->model.set_query_point(tao_to_eigen(x));
    
    eigen_to_tao(hessian, cb_data->model.hessian());
    
    return 0;
}

TaoSolver::TaoSolver() {
    TaoCreate(PETSC_COMM_SELF, &tao_);
    TaoSetType(tao_, TAONTR);
    // TaoSetTolerances(tao_, 0.0f, 1e-6f, 0.0f);
}

TaoSolver::~TaoSolver() {
    std::cout << "Destructor called \n";
    
    TaoDestroy(&tao_);
}

void TaoSolver::internal_solve(int max_iterations,
                               EnergyModel &model,
                               Eigen::VectorXf &guess) {
    int ndof = model.degrees_of_freedom();
    
    VecCreateSeq(PETSC_COMM_SELF, ndof, &tao_points_); 
    MatCreateSeqAIJ(PETSC_COMM_SELF, ndof, ndof, 300, nullptr, &tao_hessian_); 

    eigen_to_tao(tao_points_, guess);
    TaoSetInitialVector(tao_, tao_points_);

    TaoCallbackData cb_data{model};

    /* Set routines for function, gradient, hessian evaluation */
    TaoSetObjectiveAndGradientRoutine(tao_, tao_energy_gradient_callback, reinterpret_cast<void*>(&cb_data)); 
    TaoSetHessianRoutine(tao_, tao_hessian_, tao_hessian_, tao_hessian_callback, reinterpret_cast<void*>(&cb_data)); 

    TaoSetMaximumIterations(tao_, max_iterations);
    // TaoSetMaximumFunctionEvaluations(tao_, max_iterations);
    
    TaoSolve(tao_);

    TaoConvergedReason reason;
    TaoGetConvergedReason(tao_, &reason);

    ImGui::Text("Tao reason : %s", TaoConvergedReasons[reason]);
    
    guess = tao_to_eigen(tao_points_);

    VecDestroy(&tao_points_); 
    MatDestroy(&tao_hessian_);
}

void TaoSolver::improve(EnergyModel& model, Eigen::VectorXf& guess) {
    internal_solve(50, model, guess);
}

void TaoSolver::solve(EnergyModel& model, Eigen::VectorXf& guess) {
    internal_solve(INT_MAX, model, guess);
}
