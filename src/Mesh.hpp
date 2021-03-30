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

class TetraMesh {
private:
    Points points_;
    Indices<4> indices_;
    Indices<4> neighbors_;

    size_t surface_points_;

public:
    TetraMesh(const Mesh& mesh);

    const Points& points();
    PointsBlock surface_points();
    ConstPointsBlock surface_points() const;
};


