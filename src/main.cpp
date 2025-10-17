#include "Types.hpp"
#include "Mesh.hpp"
#include "VTKWriter.hpp"
#include "OpenCLSolver.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

int main() {
    // Параметры сетки и модели (перенёс сюда инициализацию)
    int_t nx = 200;
    int_t ny = 200;
    Mesh mesh(nx, ny, Float3(0.0f, 0.0f, 0.0f), Float3(10.0f, 10.0f, 0.0f));
    mesh.data.init(nx * ny);

    float_t alpha = 0.1f; // коэффициент теплопроводности

    // Расчёт шагов сетки
    float_t hx = mesh.get_hx();
    float_t hy = mesh.get_hy();

    // Вычисляем dt по условию устойчивости явной схемы (2D):
    // dt_max = 1 / (2*alpha*(1/hx^2 + 1/hy^2))
    float_t inv = (1.0f/(hx*hx) + 1.0f/(hy*hy));
    float_t dt_max = 1.0f / (2.0f * alpha * inv);
    float_t dt = 0.5f * dt_max; // запас по устойчивости

    std::cout << "nx=" << nx << " ny=" << ny << " hx=" << hx << " hy=" << hy << "\n";
    std::cout << "alpha=" << alpha << " dt=" << dt << " dt_max=" << dt_max << "\n";

    // Инициализация: две гауссовские "точки" — теперь в main
    float_t T0 = 100.0f;
    float_t sigma = 0.8f;

    Float3 vmin = mesh.get_vmin();
    Float3 vmax = mesh.get_vmax();

    float_t x1 = vmin.x + (vmax.x - vmin.x) * 0.3f;
    float_t y1 = vmin.y + (vmax.y - vmin.y) * 0.3f;
    float_t x2 = vmin.x + (vmax.x - vmin.x) * 0.7f;
    float_t y2 = vmin.y + (vmax.y - vmin.y) * 0.7f;

    float_t hx_local = mesh.get_hx();
    float_t hy_local = mesh.get_hy();

    for (int_t j = 0; j < ny; ++j) {
        for (int_t i = 0; i < nx; ++i) {
            float_t x = vmin.x + (i + 0.5f) * hx_local;
            float_t y = vmin.y + (j + 0.5f) * hy_local;
            float_t r2_1 = (x - x1)*(x - x1) + (y - y1)*(y - y1);
            float_t r2_2 = (x - x2)*(x - x2) + (y - y2)*(y - y2);
            float_t g1 = T0 * std::exp(-r2_1 / (2.0f * sigma * sigma));
            float_t g2 = T0 * std::exp(-r2_2 / (2.0f * sigma * sigma));
            mesh.data.curr.T[mesh.idx(i,j)] = g1 + g2;
            // mesh.data.next stays zero
        }
    }

    // Настройки сохранения
    int total_steps = 400;
    int save_every = 10;
    int saved = 0;

    // Попытка инициализировать OpenCL-решатель
    OpenCLSolver ocl;
    bool use_gpu = ocl.is_available();
    if (use_gpu) {
        std::cout << "Using OpenCL device for stepping.\n";
        if (!ocl.init_buffers(&mesh)) {
            std::cerr << "Failed to init OpenCL buffers — fallback to CPU.\n";
            use_gpu = false;
        }
    } else {
        std::cout << "OpenCL not available — using CPU fallback.\n";
    }

    // Сохраняем начальное состояние как step 0
    VTKWriter::save(&mesh, saved++);
    auto t_start = std::chrono::high_resolution_clock::now();

    if (use_gpu) {
        for (int step = 1; step <= total_steps; ++step) {
            bool ok = ocl.step_device(nx, ny, hx_local, hy_local, alpha, dt);
            if (!ok) {
                std::cerr << "OpenCL step failed at step " << step << " — aborting\n";
                break;
            }
            if (step % save_every == 0) {
                // считать текущее поле на хост и сохранить
                ocl.read_current_to_host(&mesh);
                VTKWriter::save(&mesh, saved++);
            }
        }
    } else {
        // CPU fallback: явная схема с периодическими BC
        for (int step = 1; step <= total_steps; ++step) {
            int_t nx_ = nx, ny_ = ny;
            float_t* Tcurr = mesh.get_T_curr();
            float_t* Tnext = mesh.get_T_next();

            for (int_t j = 0; j < ny_; ++j) {
                for (int_t i = 0; i < nx_; ++i) {
                    int_t ip = (i + 1) % nx_;
                    int_t im = (i - 1 + nx_) % nx_;
                    int_t jp = (j + 1) % ny_;
                    int_t jm = (j - 1 + ny_) % ny_;

                    int_t id   = mesh.idx(i, j);
                    int_t id_ip = mesh.idx(ip, j);
                    int_t id_im = mesh.idx(im, j);
                    int_t id_jp = mesh.idx(i, jp);
                    int_t id_jm = mesh.idx(i, jm);

                    float_t T0  = Tcurr[id];
                    float_t Txp = Tcurr[id_ip];
                    float_t Txm = Tcurr[id_im];
                    float_t Typ = Tcurr[id_jp];
                    float_t Tym = Tcurr[id_jm];

                    float_t lap_x = (Txp - 2.0f * T0 + Txm) / (hx_local * hx_local);
                    float_t lap_y = (Typ - 2.0f * T0 + Tym) / (hy_local * hy_local);

                    Tnext[id] = T0 + alpha * dt * (lap_x + lap_y);
                }
            }
            mesh.data.swap_buffers();

            if (step % save_every == 0) {
                VTKWriter::save(&mesh, saved++);
            }
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur = t_end - t_start;
    std::cout << "Finished. Wrote " << saved << " vtr files and output.pvd\n";
    std::cout << "Compute time: " << dur.count() << " s\n";

    // создаём pvd
    VTKWriter::writePVD(saved);

    return 0;
}
