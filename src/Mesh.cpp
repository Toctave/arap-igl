#include "Mesh.hpp"

#define TETLIBRARY
#include <tetgen.h>

#include <iostream>
#include <vector>

TetraMesh::TetraMesh(const Mesh& mesh) {
    Mesh* mesh_mut = const_cast<Mesh*>(&mesh);
    
    tetgenio input;

    input.mesh_dim = 3;
    
    input.pointlist = mesh_mut->V.data();
    input.numberofpoints = mesh_mut->V.rows();

    std::vector<tetgenio::facet> facets(mesh.F.rows());
    std::vector<tetgenio::polygon> polygons(mesh.F.rows());

    for (size_t i = 0; i < mesh.F.rows(); i++) {
	polygons[i].numberofvertices = 3;
	polygons[i].vertexlist = mesh_mut->F.row(i).data();

	facets[i].polygonlist = &polygons[i];
	facets[i].numberofpolygons = 1;
	facets[i].holelist = nullptr;
	facets[i].numberofholes = 0;
    }

    input.facetlist = facets.data();
    input.numberoffacets = facets.size();

    tetgenio output;
    tetrahedralize((char*)"pq", &input, &output);

    input.initialize(); // disable automatic memory deletion

    indices_ = Indices<4>(output.numberoftetrahedra, 4);
    for (size_t tet = 0; tet < indices_.rows(); tet++) {
	for (size_t corner = 0; corner < 4; corner++) {
	    indices_(tet, corner) = output.tetrahedronlist[tet * 4 + corner];
	}
    }

    neighbors_ = Indices<4>(output.numberoftetrahedra, 4);
    for (size_t tet = 0; tet < indices_.rows(); tet++) {
	for (size_t face = 0; face < 4; face++) {
	    indices_(tet, face) = output.tetrahedronlist[tet * 4 + face];
	}
    }

    points_ = Points(output.numberofpoints, 3);
    for (size_t p = 0; p < points_.rows(); p++) {
	for (size_t coord = 0; coord < 3; coord++) {
	    points_(p, coord) = output.pointlist[p * 3 + coord];
	}
    }

    surface_points_ = mesh.F.rows();
}

const Points& TetraMesh::points() {
    return points_;
}

PointsBlock TetraMesh::surface_points() {
    return points_.block<Eigen::Dynamic, 3>(0, 0, surface_points_, 3);
}

ConstPointsBlock TetraMesh::surface_points() const {
    return points_.block<Eigen::Dynamic, 3>(0, 0, surface_points_, 3);
}
