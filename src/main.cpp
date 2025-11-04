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

int main() {
    // Параметры сетки
    int_t nx = 200, ny = 200;
    Mesh mesh(nx, ny, Float3(0.0f,0.0f,0.0f), Float3(10.0f,10.0f,0.0f));
    
    // Инициализация начального состояния
    float_t T0 = 100.0f, sigma = 0.8f;
    init_gaussians(mesh, T0, sigma);
    
    // Параметры решения
    float_t alpha = 0.1f;  // коэффициент теплопроводности
    int total_steps = 400;
    int save_every = 10;
    
    // Создание и запуск солвера
    Solver solver(alpha);
    solver.solve(mesh, total_steps, save_every);
    
    return 0;
}
