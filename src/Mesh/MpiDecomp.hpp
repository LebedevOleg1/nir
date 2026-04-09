#pragma once
#include <mpi.h>
#include <vector>

// ============================================================================
// MpiDecomp — 1D Y-axis domain decomposition.
//
// Global NX×NY grid is split into horizontal strips.
// Each MPI rank owns local_ny rows of the global domain.
// Halo rows (ghost rows 0 and ny-1) are filled via exchange_halos().
//
// Memory layout per rank:
//   row 0:             ghost_bottom  (copy from rank_below)
//   rows 1..local_ny:  real cells
//   row local_ny+1:    ghost_top     (copy from rank_above)
//
// Periodic topology: rank 0 neighbours rank (size-1).
// ============================================================================
struct MpiDecomp {
    int rank       = 0;
    int size       = 1;
    int global_nx  = 0;
    int global_ny  = 0;
    int local_ny   = 0;   // real rows on this rank (without ghost rows)
    int j_start    = 0;   // global j-index of first real row

    int rank_below = -1;  // neighbour with smaller j
    int rank_above = -1;  // neighbour with larger j

    MPI_Comm comm = MPI_COMM_WORLD;

    // Split NY rows evenly; first (NY % size) ranks get one extra row.
    void init(int nx, int ny);

    // Exchange halo rows between neighbours.
    // U:    SoA buffer  U[var * stride + cell]
    // nx:   number of cells in X direction
    // total_ny: local_ny + 2 (including ghost rows)
    // nvar: number of variables
    // ncells_total: stride for SoA (may include BC ghost cells)
    void exchange_halos(float* U, int nx, int total_ny,
                        int nvar = 1, int ncells_total = 0);

    void finalize();
};
