#include "Types.hpp"
#include "Mesh.hpp"
#include "Solver.hpp"
#include <iostream>
#include <cmath>

// Инициализация температурного поля
void init_gaussians(Mesh &mesh, float_t T0, float_t sigma) {
    int_t ncells = mesh.get_ncells();
    Float3 vmin = mesh.get_vmin();
    Float3 vmax = mesh.get_vmax();

    float_t x1 = vmin.x + (vmax.x - vmin.x) * 0.3f;
    float_t y1 = vmin.y + (vmax.y - vmin.y) * 0.3f;
    float_t x2 = vmin.x + (vmax.x - vmin.x) * 0.7f;
    float_t y2 = vmin.y + (vmax.y - vmin.y) * 0.7f;

    for (int_t c = 0; c < ncells; ++c) {
        Float3 cen = mesh.centers[c];
        float_t x = cen.x;
        float_t y = cen.y;
        float_t r2_1 = (x - x1)*(x - x1) + (y - y1)*(y - y1);
        float_t r2_2 = (x - x2)*(x - x2) + (y - y2)*(y - y2);
        float_t g1 = T0 * std::exp(-r2_1 / (2.0f * sigma * sigma));
        float_t g2 = T0 * std::exp(-r2_2 / (2.0f * sigma * sigma));
        mesh.data.curr.T[c] = g1 + g2;
        mesh.data.next.T[c] = 0.0f;
    }
}

// Функция для обновления kappa на гранях
void update_face_kappa(Mesh &mesh) {
    for (int_t c = 0; c < mesh.get_ncells(); ++c) {
        for (int k = 0; k < 4; ++k) {
            int_t fi = mesh.face_index(c, k);
            int_t nb = mesh.faces[fi].neighbor;
            
            // Гармоническое среднее
            float_t d = mesh.faces[fi].distance;
            float_t d1 = d / 2.0f;
            float_t d2 = d / 2.0f;
            mesh.kappa_face[fi] = (d1 + d2) / (d1/mesh.kappa[c] + d2/mesh.kappa[nb]);
        }
    }
}

// Неоднородный материал (слоистая структура)
void init_layered_material(Mesh &mesh) {
    int_t nx = mesh.get_nx();
    int_t ny = mesh.get_ny();
    
    for (int_t j = 0; j < ny; ++j) {
        for (int_t i = 0; i < nx; ++i) {
            int_t c = mesh.idx(i, j);
            float_t x = mesh.centers[c].x;
            
            // Три слоя с разной теплопроводностью
            if (x < 3.33f) {
                mesh.kappa[c] = 0.1f;  // низкая теплопроводность
            } else if (x < 6.66f) {
                mesh.kappa[c] = 1.0f;  // средняя
            } else {
                mesh.kappa[c] = 0.1f;  // снова низкая
            }
        }
    }
    
    update_face_kappa(mesh);
}

// Источник тепла (нагреватель типа)
void add_constant_heat_source(Mesh &mesh, float_t power) {
    Float3 source_pos(5.0f, 5.0f, 0.0f);
    float_t radius = 0.5f;
    
    for (int_t c = 0; c < mesh.get_ncells(); ++c) {
        Float3 pos = mesh.centers[c];
        float_t dist_sq = (pos.x - source_pos.x)*(pos.x - source_pos.x) + 
                         (pos.y - source_pos.y)*(pos.y - source_pos.y);
        
        if (dist_sq < radius * radius) {
            mesh.source[c] = power;  // Постоянная мощность
        }
    }
}

// В main():
int main() {
    int_t nx = 200, ny = 200;
    Mesh mesh(nx, ny, Float3(0.0f,0.0f,0.0f), Float3(10.0f,10.0f,0.0f));
    
    // Инициализация с неоднородностью
    init_gaussians(mesh, 100.0f, 0.8f);
    init_layered_material(mesh);  // добавляем слои с разной теплопроводностью
    
    add_constant_heat_source(mesh, 5.0f);
    
    Solver solver(1.0f);
    solver.solve(mesh, 400, 10);
    
    return 0;
}
