#include "Mesh.hpp"
#include <cmath>

Mesh::Mesh(int_t nx_, int_t ny_, Float3 min, Float3 max, bool mpi_mode, const BCSpec bc[4])
    : nx(nx_), real_ny(ny_), v_min(min), v_max(max), mpi_mode_(mpi_mode)
{
    // Store BC specs (default: all periodic)
    for (int i = 0; i < 4; ++i)
        bc_specs[i] = bc ? bc[i] : BCSpec{BCType::Periodic, 0.0f};

    // In MPI mode: add 2 ghost rows for halo exchange
    ny = mpi_mode ? (real_ny + 2) : real_ny;
    ncells = nx * ny;

    // hy from real rows (not including MPI ghost rows)
    hx = (v_max.x - v_min.x) / static_cast<float_t>(nx);
    hy = (v_max.y - v_min.y) / static_cast<float_t>(real_ny);

    // --- Count boundary ghost cells ---
    // Ghost cells are needed for non-periodic boundaries
    n_ghost_bc = 0;
    bool need_ghost_left   = (bc_specs[(int)Boundary::Left].type   != BCType::Periodic);
    bool need_ghost_right  = (bc_specs[(int)Boundary::Right].type  != BCType::Periodic);
    bool need_ghost_bottom = (bc_specs[(int)Boundary::Bottom].type != BCType::Periodic);
    bool need_ghost_top    = (bc_specs[(int)Boundary::Top].type    != BCType::Periodic);

    // In MPI mode, Y boundaries are handled by MPI (not BC ghost cells)
    // Exception: first rank (j_start=0) might have physical bottom BC,
    // last rank might have physical top BC — but this is handled by the solver
    // setting ghost row values instead of MPI exchange. For now, in MPI mode
    // we don't create Y-direction BC ghost cells (MPI handles it).
    if (mpi_mode) {
        need_ghost_bottom = false;
        need_ghost_top = false;
    }

    if (need_ghost_left)   n_ghost_bc += ny;
    if (need_ghost_right)  n_ghost_bc += ny;
    if (need_ghost_bottom) n_ghost_bc += nx;
    if (need_ghost_top)    n_ghost_bc += nx;

    ncells_total = ncells + n_ghost_bc;

    // --- Allocate geometry ---
    centers.resize(ncells_total);
    volumes.resize(ncells_total);
    faces.resize(4 * ncells);
    cell_faces.resize(4 * ncells);
    face_boundary_id.assign(4 * ncells, -1);  // -1 = interior
    ghost_interior_map.resize(n_ghost_bc);

    // --- Cell centers and volumes ---
    for (int_t j = 0; j < ny; ++j) {
        for (int_t i = 0; i < nx; ++i) {
            int_t c = cell_index(i, j);
            float_t cy;
            if (mpi_mode) {
                cy = v_min.y + (j - 1 + 0.5f) * hy;
            } else {
                cy = v_min.y + (j + 0.5f) * hy;
            }
            centers[c] = Float3(v_min.x + (i + 0.5f) * hx, cy, 0.0f);
            volumes[c] = hx * hy;
        }
    }

    // Ghost cell centers and volumes (approximate, for distance calc)
    int_t ghost_idx = ncells;
    auto add_ghost = [&](int_t interior_cell, Float3 ghost_center) -> int_t {
        int_t gi = ghost_idx++;
        centers[gi] = ghost_center;
        volumes[gi] = hx * hy;
        ghost_interior_map[gi - ncells] = interior_cell;
        return gi;
    };

    // Track ghost cell indices for each boundary
    // ghost_left[j], ghost_right[j], ghost_bottom[i], ghost_top[i]
    std::vector<int_t> ghost_left(ny, -1), ghost_right(ny, -1);
    std::vector<int_t> ghost_bottom(nx, -1), ghost_top(nx, -1);

    if (need_ghost_left) {
        for (int_t j = 0; j < ny; ++j) {
            int_t interior = cell_index(0, j);
            Float3 gc = centers[interior];
            gc.x -= hx;
            ghost_left[j] = add_ghost(interior, gc);
        }
    }
    if (need_ghost_right) {
        for (int_t j = 0; j < ny; ++j) {
            int_t interior = cell_index(nx - 1, j);
            Float3 gc = centers[interior];
            gc.x += hx;
            ghost_right[j] = add_ghost(interior, gc);
        }
    }
    if (need_ghost_bottom) {
        for (int_t i = 0; i < nx; ++i) {
            int_t interior = cell_index(i, 0);
            Float3 gc = centers[interior];
            gc.y -= hy;
            ghost_bottom[i] = add_ghost(interior, gc);
        }
    }
    if (need_ghost_top) {
        for (int_t i = 0; i < nx; ++i) {
            int_t interior = cell_index(i, ny - 1);
            Float3 gc = centers[interior];
            gc.y += hy;
            ghost_top[i] = add_ghost(interior, gc);
        }
    }

    // --- Build faces ---
    for (int_t j = 0; j < ny; ++j) {
        for (int_t i = 0; i < nx; ++i) {
            int_t c = cell_index(i, j);

            // LEFT (k=0)
            {
                int k = 0;
                int_t fi = face_index(c, k);
                int_t nb;
                if (i == 0 && need_ghost_left) {
                    nb = ghost_left[j];
                    face_boundary_id[fi] = (int)Boundary::Left;
                } else if (i == 0 && bc_specs[(int)Boundary::Left].type == BCType::Periodic) {
                    nb = cell_index(nx - 1, j);
                } else {
                    nb = cell_index(i - 1, j);
                }
                faces.owner[fi]    = c;
                faces.neighbor[fi] = nb;
                faces.area[fi]     = hy;
                faces.normal_x[fi] = -1.0f;
                faces.normal_y[fi] = 0.0f;
                faces.distance[fi] = hx;
                cell_faces[c*4 + k] = fi;
            }

            // RIGHT (k=1)
            {
                int k = 1;
                int_t fi = face_index(c, k);
                int_t nb;
                if (i == nx - 1 && need_ghost_right) {
                    nb = ghost_right[j];
                    face_boundary_id[fi] = (int)Boundary::Right;
                } else if (i == nx - 1 && bc_specs[(int)Boundary::Right].type == BCType::Periodic) {
                    nb = cell_index(0, j);
                } else {
                    nb = cell_index(i + 1, j);
                }
                faces.owner[fi]    = c;
                faces.neighbor[fi] = nb;
                faces.area[fi]     = hy;
                faces.normal_x[fi] = 1.0f;
                faces.normal_y[fi] = 0.0f;
                faces.distance[fi] = hx;
                cell_faces[c*4 + k] = fi;
            }

            // BOTTOM (k=2)
            {
                int k = 2;
                int_t fi = face_index(c, k);
                int_t nb;
                if (mpi_mode) {
                    // MPI ghost rows: j=0 ghost→self, real→j-1
                    nb = (j == 0) ? c : cell_index(i, j - 1);
                } else if (j == 0 && need_ghost_bottom) {
                    nb = ghost_bottom[i];
                    face_boundary_id[fi] = (int)Boundary::Bottom;
                } else if (j == 0 && bc_specs[(int)Boundary::Bottom].type == BCType::Periodic) {
                    nb = cell_index(i, ny - 1);
                } else {
                    nb = cell_index(i, j - 1);
                }
                faces.owner[fi]    = c;
                faces.neighbor[fi] = nb;
                faces.area[fi]     = hx;
                faces.normal_x[fi] = 0.0f;
                faces.normal_y[fi] = -1.0f;
                faces.distance[fi] = hy;
                cell_faces[c*4 + k] = fi;
            }

            // TOP (k=3)
            {
                int k = 3;
                int_t fi = face_index(c, k);
                int_t nb;
                if (mpi_mode) {
                    nb = (j == ny - 1) ? c : cell_index(i, j + 1);
                } else if (j == ny - 1 && need_ghost_top) {
                    nb = ghost_top[i];
                    face_boundary_id[fi] = (int)Boundary::Top;
                } else if (j == ny - 1 && bc_specs[(int)Boundary::Top].type == BCType::Periodic) {
                    nb = cell_index(i, 0);
                } else {
                    nb = cell_index(i, j + 1);
                }
                faces.owner[fi]    = c;
                faces.neighbor[fi] = nb;
                faces.area[fi]     = hx;
                faces.normal_x[fi] = 0.0f;
                faces.normal_y[fi] = 1.0f;
                faces.distance[fi] = hy;
                cell_faces[c*4 + k] = fi;
            }
        }
    }
}
