#pragma once
#include "State.hpp"
#include <vector>

// Класс Mesh: хранит ячейки как список
class Mesh {
public:
    struct Face {
        int_t owner;     // индекс ячейки, к которой "приписана" грань
        int_t neighbor;  // индекс соседней ячейки (для периодики всегда валиден)
        float_t area;    // площадь грани (в 2D — длина)
        Float3 normal;   // внешняя нормаль относительно owner (единичный вектор)
        Float3 centroid; // координата центра грани
        float_t distance;// расстояние между центрами owner и neighbor вдоль нормали (positive)
    };

private:
    int_t nx, ny;
    int_t ncells;
    float_t hx, hy;
    Float3 v_min, v_max;

public:
    // cell data
    std::vector<Float3> centers;   // size = ncells
    std::vector<float_t> volumes;  // size = ncells

    // храним 4 faces на cell (size = 4*ncells)
    std::vector<Face> faces;       // size = 4*ncells

    // для быстрого доступа: для каждой ячейки индексы 4-х граней в массиве faces
    // cell_faces[cell*4 + k] = индекс в faces
    std::vector<int_t> cell_faces; // size = 4*ncells

    ExtState data; // состояние (curr/next)

    Mesh(int_t nx_, int_t ny_, Float3 min, Float3 max);
    ~Mesh() = default;

    // геттеры
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
