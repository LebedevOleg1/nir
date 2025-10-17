#include "Mesh.hpp"

Mesh::Mesh(int nx, int ny, Float3 min, Float3 max) 
    : nx(nx), ny(ny), v_min(min), v_max(max) {
    hx = (max.x - min.x) / nx;
    hy = (max.y - min.y) / ny;
    data = new ExtState(nx * ny);
}

Mesh::~Mesh() {
    delete data;
}
