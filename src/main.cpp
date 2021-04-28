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

#include "arap.hpp"
#include "arap_newton.hpp"
#include "tao_newton.hpp"

using igl::opengl::glfw::Viewer;

bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

FixedVertex closest_point(Eigen::Vector2f mouse,
			  const Mesh& mesh, const std::vector<FixedVertex>& group,
			  Eigen::Matrix4f view, Eigen::Matrix4f proj,
			  Eigen::Vector2f point_size) {
    float closest = -1;
    FixedVertex chosen = {-1, 0};

    float threshold = point_size.squaredNorm();

    Eigen::Vector4f p;
    p(3) = 1;
    for (int i = 0; i < group.size(); i++) {
	p.block<3, 1>(0, 0) = mesh.V.row(group[i].index).cast<float>();
	Eigen::Vector4f projected = (proj * view * p);
	projected /= projected(3);

	// Eigen::Vector2f projected_pixels = (projected.block<2, 1>(0, 0).array() + 1.) * viewport.block<2, 1>(2, 0).array() / 2.;
	
	if ((projected.block<2, 1>(0, 0) - mouse).squaredNorm() <= threshold
	    && (closest < 0 || projected(2) < closest)) {
	    closest = projected(2);
	    chosen = group[i];
	}
    }

    return chosen;
}

void update_group(const Eigen::MatrixXd& V, const std::vector<FixedVertex>& group, Eigen::MatrixXd& group_pos) {
    for (int i = 0; i < group.size(); i++) {
	group_pos.row(i) = V.row(group[i].index);
    }
}

Eigen::Vector2f mouse_position(const Viewer& viewer) {
    Eigen::Vector2f dimensions = viewer.core().viewport.block<2, 1>(2, 0);
    Eigen::Vector2f mouse_pos(
	viewer.current_mouse_x,
	viewer.core().viewport(3) - viewer.current_mouse_y
	);

    mouse_pos.array() = 2. * mouse_pos.array() / dimensions.array() - 1.;

    return mouse_pos;
}

Eigen::Vector4f unproject_mouse(const Viewer& viewer, Eigen::Vector3f point) {
    Eigen::Vector2f mouse_pos = mouse_position(viewer);
    Eigen::Matrix4f viewproj = viewer.core().proj * viewer.core().view;

    Eigen::Vector4f point_homo;
    
    point_homo.block<3, 1>(0, 0) = point.cast<float>();
    point_homo(3) = 1.;
    Eigen::Vector4f projected_sel = viewproj * point_homo;
    projected_sel /= projected_sel(3);

    Eigen::Vector4f mouse_homo(mouse_pos(0), mouse_pos(1), projected_sel(2), 1.);
    Eigen::Vector4f unprojected_mouse = viewproj.inverse() * mouse_homo;
    unprojected_mouse /= unprojected_mouse(3);

    return unprojected_mouse;
}

bool compare_by_index(const FixedVertex& v1, const FixedVertex& v2) {
    return v1.index < v2.index;
}

Eigen::Vector3d group_color(size_t g) {
    Eigen::Vector3d color;
    switch(g % 6) {
    case 0:
	color = Eigen::Vector3d(26, 153, 136);
	break;
    case 1:
	color = Eigen::Vector3d(235,86,0);
	break;
    case 2:
	color = Eigen::Vector3d(183, 36, 92);
	break;
    case 3:
	color = Eigen::Vector3d(243,183,0);
	break;
    case 4:
	color = Eigen::Vector3d(55,50,62);
	break;
    case 5:
	color = Eigen::Vector3d(52,89,149);
	break;
    }

    Eigen::Vector3d white(255, 255, 255);
    for (int i = 0; i < g / 6; i++) {
	color = white - (white - color) * .75;
    }
    return color / 255;
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

    // igl::per_vertex_normals(mesh.V, mesh.F, mesh.N);

    return true;
}

void benchmark(const std::string& model_name, int iterations) {
    using namespace std::chrono;
    Mesh mesh;
    LaplacianSystem system;

    std::vector<FixedVertex> fixed_vertices = {
	{0, 0},
	{1, 0},
	{2, 0},
	{3, 0},
    };
    
    if (!load_model(model_name, mesh)) {
	return;
    }

    system_init(system, &mesh, 0.);
    if (!system_bind(system, mesh.V, fixed_vertices, MEAN_VALUE)) {
	return;
    }
    
    auto t0 = high_resolution_clock::now();

    system_solve(system, iterations);

    auto t1 = high_resolution_clock::now();

    duration<double> elapsed(t1 - t0);

    std::cout << model_name << ", " << mesh.V.rows() << ", " << elapsed.count() / iterations << std::endl;
}

void solve_loop(LaplacianSystem* system) {
    while (true) {
	system_iterate(*system);
    }
}

int main(int argc, char** argv) {
    Tao tao = tao_init();
    
    Mesh mesh;
    if (!load_model(argv[1], mesh)) {
	std::cerr << "Could not load model." << std::endl;
	return 1;
    }

    Points V0 = mesh.V;

    LaplacianSystem system;
    system_init(system, &mesh, 0.0f);
    system_bind(system, V0, {{0, 0}}, COTANGENT);
    
    NewtonSolver newton_solver(mesh);

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
		    if (use_alternating) {
			system_iterate(system);
			iterations++;
		    }
		    if (use_newton) {
			newton_solver.step();
			newton_solver.apply(mesh);
			    
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

			newton_solver.set_points(mesh);
		    }

		    if (ImGui::Button("TAO solve")) {
			tao_solve_newton(tao, newton_solver);
			newton_solver.apply(mesh);
		    }
		    
		    ImGui::Text("Iterations : %d", iterations);
		    ImGui::Text("Energy : %f", newton_solver.energy());
		    ImGui::Text("Gradient norm : %f", newton_solver.gradient().norm());
		    ImGui::Text("Relative empirical gradient error : %f",
				(newton_solver.gradient() - newton_solver.empirical_gradient()).norm() / newton_solver.gradient().norm());
		    Eigen::SparseMatrix<float> hessian = newton_solver.hessian();
		    float hessian_norm = hessian.norm();
		    ImGui::Text("Hessian norm : %f", hessian_norm);


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

    tao_finalize(tao);
}

int main_(int argc, char *argv[])
{
    if (argc < 2) {
	std::cerr << "Usage : ./example <model file> <3 or more fixed indices, separated in groups by commas>\n";
	return 1;
    }

    Mesh mesh;
    if (!load_model(argv[1], mesh)) {
	std::cerr << "Could not load model." << std::endl;
	return 1;
    }
    Eigen::MatrixXf V0 = mesh.V;

    // TetraMesh tet(mesh);

    Viewer viewer;
    ImPlot::CreateContext();
    ImPlot::PushColormap(ImPlotColormap_Deep);

    struct {
	FixedVertex selected = {.index = -1};
	bool down = false;
	Eigen::Vector4f last_pos;
    } mouse;

    LaplacianSystem system;
    std::vector<FixedVertex> fixed_vertices;

    size_t curgrp = 0;
    for (int i = 2; i < argc; i++) {
	if (argv[i][0] == ',') {
	    curgrp++;
	} else {
	    fixed_vertices.push_back({atoi(argv[i]), curgrp});
	}
    }

    std::sort(fixed_vertices.begin(), fixed_vertices.end(), compare_by_index);

    Eigen::MatrixXd highlighted_points(fixed_vertices.size(), 3);
    Eigen::MatrixXd highlighted_colors(fixed_vertices.size(), 3);

    for (size_t i = 0; i < fixed_vertices.size(); i++) {
	highlighted_colors.row(i) = group_color(fixed_vertices[i].group);
	highlighted_points.row(i) = mesh.V.row(fixed_vertices[i].index).cast<double>();
    }
    
    float alpha = .0f;
    system_init(system, &mesh, alpha);
    float max_energy_ever = .0f;
    std::vector<float> energy_history;
    std::vector<float> iterations_history;

    viewer.callback_mouse_move = 
	[&fixed_vertices, &system, &highlighted_colors, &highlighted_points, &mouse, &iterations_history, &energy_history](igl::opengl::glfw::Viewer& viewer, int, int)->bool
	    {
		if (mouse.down && mouse.selected.index >= 0) {
		    Eigen::Vector4f unprojected_mouse = unproject_mouse(viewer, system.mesh->V.row(mouse.selected.index).cast<float>());
		    Eigen::Vector4f mouse_delta = unprojected_mouse - mouse.last_pos;
		    mouse.last_pos = unprojected_mouse;

		    for (const auto& vertex : fixed_vertices) {
			if (vertex.group == mouse.selected.group) {
			    system.mesh->V.row(vertex.index) += mouse_delta.block<3, 1>(0, 0);
			}
		    }
		    
		    system.iterations = 0;

		    // ImPlot::SetNextLineStyle(ImPlot::NextColormapColor());
		    // energy_history.clear();
		    energy_history.push_back(0.0f);

		    // iterations_history.clear();
		    // iterations_history.push_back(0.0f);
		    
		    update_group(system.mesh->V.cast<double>(), fixed_vertices, highlighted_points);
		    viewer.data().set_points(highlighted_points, highlighted_colors);

		    return true;
		}

		return false;
	    };

    viewer.callback_mouse_down = 
	[&mouse, &system, &fixed_vertices](igl::opengl::glfw::Viewer& viewer, int, int)->bool
	    {
		Eigen::Vector2f dimensions = viewer.core().viewport.block<2, 1>(2, 0);
		Eigen::Vector2f point_size = viewer.data().point_size / dimensions.array();
		
		mouse.down = true;

		FixedVertex closest = closest_point(mouse_position(viewer),
						    *system.mesh, fixed_vertices,
						    viewer.core().view,
						    viewer.core().proj,
						    point_size);
		mouse.selected = closest;

		if (closest.index >= 0) {
		    mouse.last_pos = unproject_mouse(viewer, system.mesh->V.row(mouse.selected.index).cast<float>());
		}
		
		return false;
	    };

    viewer.callback_mouse_up = 
	[&mouse](igl::opengl::glfw::Viewer& viewer, int, int)->bool
	    {
		mouse.down = false;

		return false;
	    };

    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();
    Eigen::MatrixXd colors(mesh.V.rows(), 3);
    for (int i = 0; i < colors.rows(); i++) {
	colors.row(i) = Eigen::Vector3d(170,170,170) / 255.;
    }

    WeightType weight_type = COTANGENT;
    const char* weight_type_labels[] = {
	"uniform",
	"cotangent",
	"cotangent_clamped",
	"cotangent_abs",
	"mean_value",
    };

    int iterations = 1;
    viewer.callback_pre_draw =
	[&](Viewer& viewer) -> bool
	    {
		if (ImGui::Begin("ARAP"))
		{
		    // Expose variable directly ...
		    if (ImGui::SliderFloat("SR alpha", &alpha, 0.0f, 0.01f, "%.5f", 3.0f)) {
			system.set_alpha(alpha);
		    }

		    if (ImGui::BeginCombo("Weight type", weight_type_labels[weight_type])) {
			for (int i = 0; i < WEIGHT_TYPE_MAX; i++) {
			    if (ImGui::Selectable(weight_type_labels[i], i == weight_type)) {
				weight_type = static_cast<WeightType>(i);
				
				if (!system_bind(system, V0, fixed_vertices, weight_type)) {
				    std::cerr << "Failed to bind mesh\n" << std::endl;
				    return 1;
				}
			    }
			}
			ImGui::EndCombo();
		    }

		    if (ImGui::Button("Bind")) {
			V0 = system.mesh->V;
			if (!system_bind(system, V0, fixed_vertices, weight_type)) {
			    std::cerr << "Failed to bind mesh\n" << std::endl;
			    return 1;
			}
		    }

		    ImGui::SliderInt("iterations per frame", &iterations, 1, 100);
		    ImGui::InputInt("Total iterations", &system.iterations);

		    const size_t max_history_steps = 200;
		    
		    ImPlot::SetNextPlotLimits(0.0, static_cast<double>(max_history_steps),
					      0.0, static_cast<double>(max_energy_ever),
					      ImGuiCond_Always);
		    if (ImPlot::BeginPlot("ARAP Energy")) {
			size_t steps = std::min(max_history_steps, energy_history.size());
			size_t first = energy_history.size() - steps;
			
			ImPlot::PlotLine("Total energy", &energy_history[first], steps);

			ImPlot::EndPlot();
		    }
		    
		    ImGui::End();
		}
		
		static const double target_frame_duration = .05;
		
		auto t1 = high_resolution_clock::now();
		duration<double> elapsed(t1 - t0);
		
		system_solve(system, iterations);

		float energy = system_energy(system);
		energy_history.push_back(energy);
		iterations_history.push_back(system.iterations);
		if (energy > max_energy_ever) {
		    max_energy_ever = energy;
		}

		viewer.data().set_vertices(mesh.V.cast<double>());

		Eigen::MatrixXd normals;
		Eigen::MatrixXd Vd = mesh.V.cast<double>();
		Eigen::MatrixXi F = mesh.F;
		
		igl::per_vertex_normals(Vd, F, normals);
		viewer.data().set_normals(normals.cast<double>());
		return false;
	    };


    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);    
    menu.callback_draw_viewer_menu =
	[&]()
	    {
		// Draw parent menu content
		menu.draw_viewer_menu();

		// Add new group
		return 1;
	    };    

    viewer.core().is_animating = true;
    viewer.core().background_color = Eigen::Vector4f(233,237,238, 255) / 255.0f;
    
    std::cout << argv[1] << " : " << mesh.V.rows() << " vertices.\n";
    if (!system_bind(system, V0, fixed_vertices, weight_type)) {
    	std::cerr << "Failed to bind mesh\n" << std::endl;
    	return 1;
    }

    viewer.data().set_points(highlighted_points, highlighted_colors);
    viewer.data().set_mesh(mesh.V.cast<double>(), mesh.F);
    viewer.data().set_colors(colors);
    viewer.data().set_face_based(false);

    // std::thread solver_thread(solve_loop, &system);
    
    viewer.launch();
    ImPlot::DestroyContext();
}
