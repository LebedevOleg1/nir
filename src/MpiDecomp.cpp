#include "MpiDecomp.hpp"
#include <iostream>

// ============================================================================
// MpiDecomp::init — разбиение глобальной сетки между MPI-процессами.
//
// Деление NY строк на size ранков: если NY не делится нацело, первые
// (NY % size) ранков получают на 1 строку больше. Это стандартный приём
// балансировки нагрузки.
//
// Пример: NY=1000, size=3 → ранки получают 334, 333, 333 строк.
//
// Периодические соседи: rank 0 граничит с rank (size-1), образуя тор.
// ============================================================================
void MpiDecomp::init(int nx, int ny) {
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    global_nx = nx;
    global_ny = ny;

    // Равномерное распределение строк: base + 1 для первых remainder ранков
    int base      = ny / size;
    int remainder = ny % size;

    if (rank < remainder) {
        local_ny = base + 1;
        j_start  = rank * (base + 1);
    } else {
        local_ny = base;
        j_start  = remainder * (base + 1) + (rank - remainder) * base;
    }

    // Периодические соседи (тороидальная топология)
    rank_below = (rank - 1 + size) % size;
    rank_above = (rank + 1) % size;

    if (rank == 0) {
        std::cout << "MPI: " << size << " ranks, grid " << nx << "x" << ny << "\n";
    }
    std::cout << "  rank " << rank << ": local_ny=" << local_ny
              << ", j_start=" << j_start << "\n";
}

// ============================================================================
// exchange_halos — обмен ghost-строками между соседями.
//
// Раскладка памяти T (nx * (local_ny + 2)):
//   строка 0:             ghost_bottom (от rank_below)
//   строки 1..local_ny:   реальные данные
//   строка local_ny+1:    ghost_top (от rank_above)
//
// Обмен через MPI_Sendrecv — неблокирующие попарные пересылки:
//   1) Отправляем строку 1 (нижнюю реальную) соседу снизу → его ghost_top
//      Получаем от соседа снизу его верхнюю строку → наш ghost_bottom (строка 0)
//   2) Отправляем строку local_ny (верхнюю реальную) соседу сверху → его ghost_bottom
//      Получаем от соседа сверху его нижнюю строку → наш ghost_top
//
// MPI_Sendrecv безопаснее пары Send+Recv: гарантирует отсутствие deadlock'ов.
// ============================================================================
void MpiDecomp::exchange_halos(float* U, int nx, int total_ny, int nvar, int ncells_total_param) {
    int row_size = nx;
    int local_rows = total_ny - 2;  // without ghost rows
    // For SoA layout: variable v is at U[v * stride + cell]
    // stride = ncells_total (includes all cells including BC ghosts)
    // If ncells_total_param not given, assume dense layout (nx * total_ny)
    int stride = (ncells_total_param > 0) ? ncells_total_param : (nx * total_ny);

    MPI_Status status;

    for (int v = 0; v < nvar; ++v) {
        float* base = U + v * stride;

        float* ghost_bottom    = base;
        float* first_real_row  = base + row_size;
        float* last_real_row   = base + local_rows * row_size;
        float* ghost_top       = base + (local_rows + 1) * row_size;

        // Send bottom real row down, receive from above into ghost_top
        MPI_Sendrecv(
            first_real_row, row_size, MPI_FLOAT, rank_below, 0,
            ghost_top,      row_size, MPI_FLOAT, rank_above, 0,
            comm, &status
        );

        // Send top real row up, receive from below into ghost_bottom
        MPI_Sendrecv(
            last_real_row,  row_size, MPI_FLOAT, rank_above, 1,
            ghost_bottom,   row_size, MPI_FLOAT, rank_below, 1,
            comm, &status
        );
    }
}

void MpiDecomp::finalize() {
    MPI_Finalize();
}
