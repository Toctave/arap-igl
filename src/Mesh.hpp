#pragma once

#include <Eigen/Core>

using Points = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;

using PointsBlock = Eigen::Block<Points, Eigen::Dynamic, 3>;
using ConstPointsBlock = const Eigen::Block<const Points, Eigen::Dynamic, 3>;

template<int N>
using Indices = Eigen::Matrix<int, Eigen::Dynamic, N, Eigen::RowMajor>;

struct Mesh {
    Points V;
    Indices<3> F;
};

struct TetraMesh {
    Points points;
    Indices<4> indices;
    Indices<4> neighbors;

    size_t surface_points;

    TetraMesh(const Mesh& mesh);
    Eigen::Ref<const Eigen::Vector3f> point(int tet, int index) const;
};


