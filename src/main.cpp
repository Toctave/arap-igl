#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/per_vertex_normals.h>

#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
#include <chrono>

#include <thread>

#include <implot.h>

#include "model_solver.hpp"
#include "arap_model.hpp"
#include "newton_solver.hpp"
#include "tao_solver.hpp"

using igl::opengl::glfw::Viewer;

bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}
bool load_model(const std::string& model_name, Mesh& mesh) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (hasEnding(model_name, ".off")) {
	igl::readOFF(model_name, V, F);
    } else if (hasEnding(model_name, ".obj")) {
	igl::readOBJ(model_name, V, F);
    } else {
	return false;
    }
    mesh.F = F.cast<int>();
    mesh.V = V.cast<float>();

    return true;
}

Eigen::VectorXf flat_vertex_vector(const Mesh& mesh) {
    return Eigen::Map<const Eigen::VectorXf>(mesh.V.data(), mesh.V.size());
}

void set_from_flat_vertices(Mesh& mesh, const Eigen::VectorXf& v) {
    mesh.V = Eigen::Map<const Points>(v.data(), v.size() / 3, 3);
}

int main(int argc, char** argv) {
    PetscInitialize(0,nullptr,(char*)0,0);
    
    Mesh mesh;
    if (!load_model(argv[1], mesh)) {
	std::cerr << "Could not load model." << std::endl;
	return 1;
    }

    Points V0 = mesh.V;

    ARAPModel arap(mesh);
    // NewtonSolver newton_solver(10, .4f);
    TaoSolver newton_solver;

    int iterations = 0;
    float amp = .0f;
    float freq = 3.0f;
    float rot = 0.0f;
    float trans = 0.0f;
    bool use_newton = false;
    bool use_alternating = false;
    int iterations_per_frame = 1;
    float twist = 0.0f;
    int max_iterations = 0;

    Viewer viewer;
    ImPlot::CreateContext();
    
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);    
    
    viewer.core().is_animating = true;
    viewer.data().set_mesh(mesh.V.cast<double>(), mesh.F);
    viewer.data().set_face_based(true);

    viewer.callback_pre_draw =
	[&](Viewer& viewer) -> bool
	    {
		for (int i = 0; i < iterations_per_frame; i++) {
		    if (max_iterations && iterations >= max_iterations) {
			break;
		    }
		    if (use_newton) {
			Eigen::VectorXf guess = flat_vertex_vector(mesh);
			
			newton_solver.improve(arap, guess);

			set_from_flat_vertices(mesh, guess);
			    
			iterations++;
		    }
		}
		if (ImGui::Begin("ARAP")) {
		    bool change = ImGui::SliderFloat("Deformation amplitude", &amp, 0.0f, 1.0f);
		    change = ImGui::SliderFloat("Deformation frequency", &freq, 0.0f, 10.0f) || change;
		    change = ImGui::SliderFloat("Rotation", &rot, 0.0f, 1.0f) || change;
		    change = ImGui::SliderFloat("Translation", &trans, 0.0f, 1.0f) || change;
		    change = ImGui::SliderFloat("Twist", &twist, 0.0f, 1.0f) || change;

		    change = ImGui::Checkbox("Use newton solver", &use_newton) || change;
		    change = ImGui::Checkbox("Use alternating solver", &use_alternating) || change;
		    change = ImGui::SliderInt("Iterations per frame", &iterations_per_frame, 0, 100) || change;
		    change = ImGui::SliderInt("Max iterations", &max_iterations, 0, 100) || change;
		    
		    if (change) {
			for (int i = 0; i < mesh.V.rows(); i++) {
			    float dx = amp * std::sin(2.0f * M_PI * freq * V0(i, 1));

			    float theta = 2.0f * M_PI * rot;
			    Eigen::Matrix3f rot_mat;
			    rot_mat <<
				std::cos(theta), std::sin(theta), 0,
				-std::sin(theta), std::cos(theta), 0,
				0, 0, 1;

			    float theta_twist = 2.0f * M_PI * twist * V0(i, 1);
			    Eigen::Matrix3f rot_mat_twist;
			    rot_mat_twist <<
				std::cos(theta_twist), std::sin(theta_twist), 0,
				-std::sin(theta_twist), std::cos(theta_twist), 0,
				0, 0, 1;

			    mesh.V.row(i) =
				((V0.row(i) + Eigen::RowVector3f(dx, 0, 0))
				 * rot_mat.transpose()
				 + Eigen::RowVector3f(trans, 0.0f, 0.0f))
				* rot_mat_twist;
			    iterations = 0;
			}
		    }

		    if (ImGui::Button("Solve")) {
			Eigen::VectorXf guess = flat_vertex_vector(mesh);
			
			newton_solver.solve(arap, guess);

			set_from_flat_vertices(mesh, guess);
		    }
		    
		    // ImGui::Text("Iterations : %d", iterations);
		    // ImGui::Text("Energy : %f", arap.energy());
		    // ImGui::Text("Gradient norm : %f", arap.gradient().norm());
		    // ImGui::Text("Relative empirical gradient error : %f",
				// (newton_solver.gradient() - newton_solver.empirical_gradient()).norm() / newton_solver.gradient().norm());
		    // Eigen::SparseMatrix<float> hessian = newton_solver.hessian();
		    // float hessian_norm = hessian.norm();
		    // ImGui::Text("Hessian norm : %f", hessian_norm);

		    // Eigen::SparseMatrix<float> empirical_hessian = newton_solver.empirical_hessian();
		    // float hessian_diff = (empirical_hessian - hessian).norm();
		    // ImGui::Text("Relative empirical hessian error : %f", hessian_diff / hessian_norm);

		    ImGui::End();
		}
		
		Eigen::RowVector3f avg = mesh.V.colwise().sum() / mesh.V.rows();
		mesh.V.rowwise() -= avg;
		
		viewer.data().set_vertices(mesh.V.cast<double>());
		
		return false;
	    };
    
    menu.callback_draw_viewer_menu =
	[&]()
	    {
		// Draw parent menu content
		menu.draw_viewer_menu();

		// Add new group
		return 1;
	    };    
    

    viewer.launch();
    ImPlot::DestroyContext();

    PetscFinalize();
}
