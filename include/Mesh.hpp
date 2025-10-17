#pragma once
#include "State.hpp"

// Класс, содержащий геометрию сетки и состояния
class Mesh {
private:
    int nx, ny;
    Float3 v_min, v_max;
    float hx, hy;

public:
    ExtState* data;

    Mesh(int nx, int ny, Float3 min, Float3 max);
    ~Mesh();

    // Геттеры
    Float3 get_vmin() const { return v_min; }
    Float3 get_vmax() const { return v_max; }
    float* get_T_curr() { return data->curr->T; }
    float* get_T_next() { return data->next->T; }
    int get_nx() const { return nx; }
    int get_ny() const { return ny; }
    float get_hx() const { return hx; }
    float get_hy() const { return hy; }

    // индекс в линейном массиве по (i,j)
    inline int idx(int i, int j) const { return j * nx + i; }
};
