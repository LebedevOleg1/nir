#include "Solver.hpp"
#include <cmath>
#include <algorithm>

void Solver::calculate_dt() {
    float hx = mesh->get_hx();
    float hy = mesh->get_hy();
    // Условие устойчивости явной схемы (2D):
    // dt_max = 1 / (2*alpha*(1/hx^2 + 1/hy^2))
    float inv = (1.0f/(hx*hx) + 1.0f/(hy*hy));
    float dt_max = 1.0f / (2.0f * alpha * inv);
    dt = 0.5f * dt_max; // запас по устойчивости
}

void Solver::apply_bc(float* T) {
    // Дирихле: границы = 0
    const int nx = mesh->get_nx();
    const int ny = mesh->get_ny();

    // нижняя и верхняя строки
    for (int i = 0; i < nx; ++i) {
        T[mesh->idx(i, 0)] = 0.0f;
        T[mesh->idx(i, ny-1)] = 0.0f;
    }
    // левая и правая колонки
    for (int j = 0; j < ny; ++j) {
        T[mesh->idx(0, j)] = 0.0f;
        T[mesh->idx(nx-1, j)] = 0.0f;
    }
}

void Solver::initialize(float T0, float sigma) {
    const int nx = mesh->get_nx();
    const int ny = mesh->get_ny();
    float* T = mesh->get_T_curr();

    float x_center = (mesh->get_vmax().x + mesh->get_vmin().x) / 2.0f;
    float y_center = (mesh->get_vmax().y + mesh->get_vmin().y) / 2.0f;
    float hx = mesh->get_hx();
    float hy = mesh->get_hy();
    Float3 vmin = mesh->get_vmin();
    Float3 vmax = mesh->get_vmax();

    // Две гауссовские "точки" — расположим одну в 30% от границы, другую в 70%
    float x1 = vmin.x + (vmax.x - vmin.x) * 0.3f;
    float y1 = vmin.y + (vmax.y - vmin.y) * 0.3f;
    float x2 = vmin.x + (vmax.x - vmin.x) * 0.7f;
    float y2 = vmin.y + (vmax.y - vmin.y) * 0.7f;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            float x = vmin.x + (i + 0.5f) * hx;
            float y = vmin.y + (j + 0.5f) * hy;
            float r2_1 = (x - x1)*(x - x1) + (y - y1)*(y - y1);
            float r2_2 = (x - x2)*(x - x2) + (y - y2)*(y - y2);
            float g1 = T0 * std::exp(-r2_1 / (2.0f * sigma * sigma));
            float g2 = T0 * std::exp(-r2_2 / (2.0f * sigma * sigma));
            T[mesh->idx(i,j)] = g1 + g2;
        }
    }

    // применим BC к начальному
    apply_bc(T);
}

void CpuSolver::step() {
    const int nx = mesh->get_nx();
    const int ny = mesh->get_ny();
    const float hx = mesh->get_hx();
    const float hy = mesh->get_hy();
    float* T_curr = mesh->get_T_curr();
    float* T_next = mesh->get_T_next();

    // предварительно нулём next (не обязательно, но безопасно)
    // for (int i=0;i<nx*ny;++i) T_next[i] = 0.0f;

    // внутренние узлы
    for (int i = 0; i < n_cells; ++i) {
        Tcurr left_idx[i];
    }
    for (int j = 1; j < ny-1; ++j) {
        for (int i = 1; i < nx-1; ++i) {
            int id = mesh->idx(i,j);

            float Txp = T_curr[mesh->idx(i+1, j)];
            float Txm = T_curr[mesh->idx(i-1, j)];
            float Typ = T_curr[mesh->idx(i, j+1)];
            float Tym = T_curr[mesh->idx(i, j-1)];
            float T0  = T_curr[id];

            float lap_x = (Txp - 2.0f*T0 + Txm) / (hx*hx);
            float lap_y = (Typ - 2.0f*T0 + Tym) / (hy*hy);

            T_next[id] = T0 + alpha * dt * (lap_x + lap_y);
        }
    }

    // граничные значения на next
    apply_bc(T_next);

    // обмен буферов
    mesh->data->swap();
}
