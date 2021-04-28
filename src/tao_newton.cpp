#include "tao_newton.hpp"

#include <iostream>

#include "arap_newton.hpp"

void eigen_to_tao(float* tao, float eigen) {
    *tao = eigen;
}

void eigen_to_tao(Vec tao, const Eigen::VectorXf& eigen) {
    std::vector<int> indices(eigen.size());
    for (int i = 0; i < eigen.size(); i++) {
	indices[i] = i;
    }
    
    VecSetValues(tao, indices.size(), indices.data(), eigen.data(), INSERT_VALUES);
    VecAssemblyBegin(tao);
    VecAssemblyEnd(tao);
}

void eigen_to_tao(Mat tao, const Eigen::SparseMatrix<float>& eigen) {
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

Eigen::VectorXf tao_to_eigen(Vec tao) {
    int size;
    VecGetSize(tao, &size);
    
    const float* data;
    VecGetArrayRead(tao, &data);

    Eigen::VectorXf rval = Eigen::Map<const Eigen::VectorXf>(data, size);

    return rval;
}

PetscErrorCode compute_energy_and_gradient(Tao tao, Vec x, PetscReal* energy, Vec gradient, void* user) {
    NewtonSolver* solver = reinterpret_cast<NewtonSolver*>(user);

    solver->set_points(tao_to_eigen(x));

    eigen_to_tao(energy, solver->energy());
    eigen_to_tao(gradient, solver->gradient());

    return 0;
}

PetscErrorCode compute_hessian(Tao tao, Vec x, Mat hessian, Mat preconditioner, void* user) {
    NewtonSolver* solver = reinterpret_cast<NewtonSolver*>(user);

    solver->set_points(tao_to_eigen(x));
    
    eigen_to_tao(hessian, solver->hessian());
    
    return 0;
}

Tao tao_init() {
    Tao tao;
    PetscInitialize(0,nullptr,(char*)0,0);
    TaoCreate(PETSC_COMM_SELF, &tao);

    return tao;
}

void tao_finalize(Tao tao) {
    TaoDestroy(&tao);
    PetscFinalize();
}

void tao_solve_newton(Tao tao, NewtonSolver& solver) {
    Vec x;
    Mat hessian;

    PetscErrorCode ierr;

    ierr = VecCreateSeq(PETSC_COMM_SELF, solver.ndof(), &x); 
    ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF, 2, solver.ndof(), solver.ndof(), 1, NULL, &hessian); 

    /* Create TAO solver with desired solution method */
    ierr = TaoCreate(PETSC_COMM_SELF,&tao);
    ierr = TaoSetType(tao,TAONTR);
  
    eigen_to_tao(x, solver.current_points());
    ierr = TaoSetInitialVector(tao,x);

    /* Set routines for function, gradient, hessian evaluation */
    ierr = TaoSetObjectiveAndGradientRoutine(tao, compute_energy_and_gradient, reinterpret_cast<void*>(&solver)); 
    ierr = TaoSetHessianRoutine(tao, hessian, hessian, compute_hessian, reinterpret_cast<void*>(&solver)); 

    ierr = TaoSolve(tao);

    Eigen::VectorXf result = tao_to_eigen(x);
    solver.set_points(result);

    ierr = VecDestroy(&x); 
    ierr = MatDestroy(&hessian);
}
