#include "Mesh.hpp"

#define TETLIBRARY
#include <tetgen.h>

#include <iostream>
#include <vector>

#include <Eigen/Geometry>

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
    tetrahedralize((char*)"pqn", &input, &output);

    input.initialize(); // disable automatic memory deletion

    indices = Indices<4>(output.numberoftetrahedra, 4);
    for (size_t tet = 0; tet < indices.rows(); tet++) {
        for (size_t corner = 0; corner < 4; corner++) {
            indices(tet, corner) = output.tetrahedronlist[tet * 4 + corner];
        }
    }

    neighbors = Indices<4>(output.numberoftetrahedra, 4);
    for (size_t tet = 0; tet < indices.rows(); tet++) {
        for (size_t face = 0; face < 4; face++) {
            neighbors(tet, face) = output.neighborlist[tet * 4 + face];
        }
    }

    points = Points(output.numberofpoints, 3);
    for (size_t p = 0; p < points.rows(); p++) {
        for (size_t coord = 0; coord < 3; coord++) {
            points(p, coord) = output.pointlist[p * 3 + coord];
        }
    }

    surface_points = mesh.F.rows();
}

Eigen::Ref<const Eigen::Vector3f> TetraMesh::point(int tet, int index) const {
    return points.row(indices(tet, index)).transpose();
}
