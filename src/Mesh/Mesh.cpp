#include "Mesh/Mesh.hpp"
#include <cmath>

Mesh::Mesh(Int nx_, Int ny_, Vec3 vmin, Vec3 vmax, bool mpi_mode, const BCSpec bc[4])
    : nx(nx_), real_ny(ny_), v_min(vmin), v_max(vmax), mpi_mode_(mpi_mode)
{
    for (int i = 0; i < 4; ++i)
        bc_specs[i] = bc ? bc[i] : BCSpec{BCType::Periodic, 0.0f};

    // In MPI mode: add 2 ghost rows (top and bottom halo)
    ny = mpi_mode ? (real_ny + 2) : real_ny;
    ncells = nx * ny;

    hx = (v_max.x - v_min.x) / Real(nx);
    hy = (v_max.y - v_min.y) / Real(real_ny);

    // --- Count BC ghost cells ---
    n_ghost_bc = 0;
    bool need_ghost_left   = (bc_specs[(int)Boundary::Left].type   != BCType::Periodic);
    bool need_ghost_right  = (bc_specs[(int)Boundary::Right].type  != BCType::Periodic);
    bool need_ghost_bottom = (bc_specs[(int)Boundary::Bottom].type != BCType::Periodic);
    bool need_ghost_top    = (bc_specs[(int)Boundary::Top].type    != BCType::Periodic);

    // In MPI mode, Y boundaries are handled by halo exchange, not BC ghosts
    if (mpi_mode) {
        need_ghost_bottom = false;
        need_ghost_top    = false;
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
    face_boundary_id.assign(4 * ncells, -1);
    ghost_interior_map.resize(n_ghost_bc);

    // --- Cell centers and volumes ---
    for (Int j = 0; j < ny; ++j) {
        for (Int i = 0; i < nx; ++i) {
            Int c = cell_index(i, j);
            Real cy;
            if (mpi_mode) {
                cy = v_min.y + (j - 1 + 0.5f) * hy;
            } else {
                cy = v_min.y + (j + 0.5f) * hy;
            }
            centers[c] = Vec3(v_min.x + (i + 0.5f) * hx, cy);
            volumes[c] = hx * hy;
        }
    }

    // --- Add BC ghost cells ---
    Int ghost_idx = ncells;
    auto add_ghost = [&](Int interior_cell, Vec3 gc) -> Int {
        Int gi = ghost_idx++;
        centers[gi] = gc;
        volumes[gi] = hx * hy;
        ghost_interior_map[gi - ncells] = interior_cell;
        return gi;
    };

    std::vector<Int> ghost_left(ny, -1), ghost_right(ny, -1);
    std::vector<Int> ghost_bottom(nx, -1), ghost_top(nx, -1);

    if (need_ghost_left) {
        for (Int j = 0; j < ny; ++j) {
            Int interior = cell_index(0, j);
            Vec3 gc = centers[interior]; gc.x -= hx;
            ghost_left[j] = add_ghost(interior, gc);
        }
    }
    if (need_ghost_right) {
        for (Int j = 0; j < ny; ++j) {
            Int interior = cell_index(nx - 1, j);
            Vec3 gc = centers[interior]; gc.x += hx;
            ghost_right[j] = add_ghost(interior, gc);
        }
    }
    if (need_ghost_bottom) {
        for (Int i = 0; i < nx; ++i) {
            Int interior = cell_index(i, 0);
            Vec3 gc = centers[interior]; gc.y -= hy;
            ghost_bottom[i] = add_ghost(interior, gc);
        }
    }
    if (need_ghost_top) {
        for (Int i = 0; i < nx; ++i) {
            Int interior = cell_index(i, ny - 1);
            Vec3 gc = centers[interior]; gc.y += hy;
            ghost_top[i] = add_ghost(interior, gc);
        }
    }

    // --- Build faces ---
    for (Int j = 0; j < ny; ++j) {
        for (Int i = 0; i < nx; ++i) {
            Int c = cell_index(i, j);

            // LEFT (k=0)
            {
                Int fi = face_index(c, 0);
                Int nb;
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
                cell_faces[c*4 + 0] = fi;
            }

            // RIGHT (k=1)
            {
                Int fi = face_index(c, 1);
                Int nb;
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
                cell_faces[c*4 + 1] = fi;
            }

            // BOTTOM (k=2)
            {
                Int fi = face_index(c, 2);
                Int nb;
                if (mpi_mode) {
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
                cell_faces[c*4 + 2] = fi;
            }

            // TOP (k=3)
            {
                Int fi = face_index(c, 3);
                Int nb;
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
                cell_faces[c*4 + 3] = fi;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Build MUSCL stencil arrays.
    // For face k of cell c: face_stencil_owner[fi]    = neighbor of c through face (k^1)
    //                       face_stencil_neighbor[fi] = neighbor of nb through face k
    // This gives the 4-cell stencil needed for 2nd-order MUSCL reconstruction.
    // Ghost cells (nb >= ncells) are used as-is — their mirrored values
    // produce zero slope at non-periodic boundaries, giving 1st-order at walls.
    // -----------------------------------------------------------------------
    face_stencil_owner.resize(4 * ncells);
    face_stencil_neighbor.resize(4 * ncells);

    for (Int c = 0; c < ncells; ++c) {
        for (int k = 0; k < 4; ++k) {
            Int fi = cell_faces[c * 4 + k];
            Int nb = faces.neighbor[fi];

            // stencil_owner: neighbor of c through opposite face (k^1 flips 0↔1, 2↔3)
            face_stencil_owner[fi] = faces.neighbor[cell_faces[c * 4 + (k ^ 1)]];

            // stencil_neighbor: neighbor of nb through face k
            // (extends stencil one more cell beyond nb)
            if (nb < ncells) {
                face_stencil_neighbor[fi] = faces.neighbor[cell_faces[nb * 4 + k]];
            } else {
                // Ghost cell: no cell_faces entry; use nb itself → zero slope
                face_stencil_neighbor[fi] = nb;
            }
        }
    }
}
