#pragma once
#include "State.hpp"
#include <vector>

// ============================================================================
// Mesh — конечно-объёмная сетка (FVM) для 2D прямоугольной области.
//
// Грани хранятся в формате SoA (Structure of Arrays): каждое свойство грани
// (owner, neighbor, area, ...) лежит в своём непрерывном массиве.
// Это критично для GPU — потоки warp'а читают соседние элементы одного
// массива, а не разбросанные поля структуры. Результат: coalesced memory
// access, который в ~10x быстрее random access на GPU.
//
// Топология: каждая ячейка имеет ровно 4 грани (left, right, bottom, top).
// cell_faces[cell*4 + k] даёт индекс k-й грани ячейки в массивах faces.
// ============================================================================
class Mesh {
public:
    // SoA-хранилище граней — «Structure of Arrays».
    // Вместо vector<Face> (AoS) храним отдельные массивы для каждого поля.
    // При обходе по одному полю (например, все owner) данные лежат подряд
    // в памяти → кэш-линии используются на 100%, а на GPU каждый warp
    // загружает один непрерывный блок.
    struct Faces {
        std::vector<int_t>   owner;     // индекс ячейки-владельца грани
        std::vector<int_t>   neighbor;  // индекс соседней ячейки через грань
        std::vector<float_t> area;      // площадь грани (в 2D — длина ребра)
        std::vector<Float3>  normal;    // единичная внешняя нормаль
        std::vector<Float3>  centroid;  // координаты центра грани
        std::vector<float_t> distance;  // расстояние между центрами owner и neighbor
        int_t count = 0;               // количество граней

        void resize(int_t n) {
            count = n;
            owner.resize(n);
            neighbor.resize(n);
            area.resize(n);
            normal.resize(n);
            centroid.resize(n);
            distance.resize(n);
        }
    };

private:
    int_t nx, ny;
    int_t ncells;
    float_t hx, hy;
    Float3 v_min, v_max;

public:
    std::vector<Float3> centers;       // координаты центров ячеек
    std::vector<float_t> volumes;      // объёмы ячеек (в 2D — площади)

    std::vector<float_t> kappa;        // теплопроводность ячеек
    std::vector<float_t> kappa_face;   // теплопроводность на гранях (гармоническое среднее)
    std::vector<float_t> source;       // источниковый член q(x,y,t)

    Faces faces;                       // SoA-массивы граней

    std::vector<int_t> cell_faces;     // cell_faces[cell*4+k] = индекс k-й грани

    ExtState data;                     // двойной буфер температуры (curr/next)

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
