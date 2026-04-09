#include "Mesh/MpiDecomp.hpp"
#include <iostream>

void MpiDecomp::init(int nx, int ny) {
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    global_nx = nx;
    global_ny = ny;

    // Even load balancing: first (ny % size) ranks get one extra row
    int base      = ny / size;
    int remainder = ny % size;

    if (rank < remainder) {
        local_ny = base + 1;
        j_start  = rank * (base + 1);
    } else {
        local_ny = base;
        j_start  = remainder * (base + 1) + (rank - remainder) * base;
    }

    // Periodic neighbours (toroidal topology)
    rank_below = (rank - 1 + size) % size;
    rank_above = (rank + 1) % size;

    if (rank == 0) {
        std::cout << "MPI: " << size << " ranks, grid " << nx << "x" << ny << "\n";
    }
    std::cout << "  rank " << rank << ": local_ny=" << local_ny
              << ", j_start=" << j_start << "\n";
}

// ============================================================================
// Halo exchange via MPI_Sendrecv (deadlock-free by design).
//
// For each variable v in SoA layout:
//   Send first_real_row → ghost_top of rank_below
//   Recv from rank_above → our ghost_top
//   Send last_real_row → ghost_bottom of rank_above
//   Recv from rank_below → our ghost_bottom
// ============================================================================
void MpiDecomp::exchange_halos(float* U, int nx, int total_ny,
                                int nvar, int ncells_total_param) {
    int row_size   = nx;
    int local_rows = total_ny - 2;
    int stride     = (ncells_total_param > 0) ? ncells_total_param : (nx * total_ny);

    MPI_Status status;

    for (int v = 0; v < nvar; ++v) {
        float* base = U + v * stride;

        float* ghost_bottom   = base;
        float* first_real_row = base + row_size;
        float* last_real_row  = base + local_rows * row_size;
        float* ghost_top      = base + (local_rows + 1) * row_size;

        // Send our first real row down; receive into ghost_top from above
        MPI_Sendrecv(
            first_real_row, row_size, MPI_FLOAT, rank_below, 0,
            ghost_top,      row_size, MPI_FLOAT, rank_above, 0,
            comm, &status
        );

        // Send our last real row up; receive into ghost_bottom from below
        MPI_Sendrecv(
            last_real_row, row_size, MPI_FLOAT, rank_above, 1,
            ghost_bottom,  row_size, MPI_FLOAT, rank_below, 1,
            comm, &status
        );
    }
}

void MpiDecomp::finalize() {
    MPI_Finalize();
}
