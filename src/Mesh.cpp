// src/Mesh.cpp
#include "Mesh.hpp"

Mesh::Mesh(int_t nx_, int_t ny_, Float3 min, Float3 max)
    : nx(nx_), ny(ny_), v_min(min), v_max(max) {
    hx = (v_max.x - v_min.x) / static_cast<float_t>(nx_);
    hy = (v_max.y - v_min.y) / static_cast<float_t>(ny_);
    data.init(static_cast<int_t>(nx_ * ny_));
}

// NOTE: no Mesh::~Mesh() definition here because in Mesh.hpp the destructor is = default;
