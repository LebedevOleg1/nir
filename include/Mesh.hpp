#pragma once
#include "State.hpp"

// Класс, содержащий геометрию сетки и состояния
class Mesh {
private:
    int_t nx, ny;
    Float3 v_min, v_max;
    float_t hx, hy;

public:
    ExtState data; // теперь объект, не указатель

    Mesh(int_t nx_, int_t ny_, Float3 min, Float3 max);
    ~Mesh() = default;

    // Геттеры
    Float3 get_vmin() const { return v_min; }
    Float3 get_vmax() const { return v_max; }
    float_t* get_T_curr() { return data.curr.T.data(); }
    float_t* get_T_next() { return data.next.T.data(); }
    int_t get_nx() const { return nx; }
    int_t get_ny() const { return ny; }
    float_t get_hx() const { return hx; }
    float_t get_hy() const { return hy; }

    // индекс в линейном массиве по (i,j)
    inline int_t idx(int_t i, int_t j) const { return j * nx + i; }
};
