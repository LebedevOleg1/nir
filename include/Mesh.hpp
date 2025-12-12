#pragma once
#include "State.hpp"
#include <vector>

class Mesh {
public:
    struct Face {
        int_t owner;
        int_t neighbor;
        float_t area;
        Float3 normal;
        Float3 centroid;
        float_t distance;
    };

private:
    int_t nx, ny;
    int_t ncells;
    float_t hx, hy;
    Float3 v_min, v_max;

public:
    std::vector<Float3> centers;
    std::vector<float_t> volumes;

    std::vector<float_t> kappa;
    std::vector<float_t> kappa_face;
    std::vector<float_t> source;

    std::vector<Face> faces;

    std::vector<int_t> cell_faces;

    ExtState data;

    Mesh(int_t nx_, int_t ny_, Float3 min, Float3 max);
    ~Mesh() = default;

    int_t get_nx() const { return nx; }
    int_t get_ny() const { return ny; }
    int_t get_ncells() const { return ncells; }
    float_t get_hx() const { return hx; }
    float_t get_hy() const { return hy; }
    Float3 get_vmin() const { return v_min; }
    Float3 get_vmax() const { return v_max; }

    float_t* get_T_curr() { return data.curr.T.data(); }
    float_t* get_T_next() { return data.next.T.data(); }

    inline int_t cell_index(int_t i, int_t j) const { return j * nx + i; }
    inline int_t idx(int_t i, int_t j) const { return cell_index(i,j); }
    inline int_t face_index(int_t cell, int k) const { return cell * 4 + k; }
};