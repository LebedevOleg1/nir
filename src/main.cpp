#include "Solver.hpp"
#include "VTKWriter.hpp"
#include <iostream>

int main() {
    // Параметры сетки и модели
    int nx = 100;
    int ny = 100;
    Mesh mesh(nx, ny, {0.0f, 0.0f, 0.0f}, {10.0f, 10.0f, 0.0f});
    float alpha = 0.1f; // коэффициент теплопроводности
    CpuSolver solver(&mesh, alpha);

    // Инициализация: амплитуда и ширина гауссов
    solver.initialize(100.0f, 0.8f);

    // Настройки сохранения
    int total_steps = 400;
    int save_every = 10;
    int saved = 0;

    // Сохраняем начальное состояние как step 0
    VTKWriter::save(&mesh, saved++);
    for (int step = 1; step <= total_steps; ++step) {
        solver.step();
        if (step % save_every == 0) {
            VTKWriter::save(&mesh, saved++);
        }
    }

    // создаём pvd, где будет saved файлов (output_0000.vtr ... output_00NN.vtr)
    VTKWriter::writePVD(saved);

    std::cout << "Finished. Wrote " << saved << " vtr files and output.pvd\n";
    return 0;
}
