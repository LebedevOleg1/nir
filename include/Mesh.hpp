#pragma once
#include "Types.hpp"
#include "BoundaryCondition.hpp"
#include <vector>

// ============================================================================
// Mesh -- 2D rectangular FVM mesh (geometry only, no physics data).
//
// Faces stored as SoA for coalesced GPU access.
// 4 faces per cell: left(0), right(1), bottom(2), top(3).
//
// Boundary conditions:
//   - Periodic: face neighbor wraps around (no ghost cells)
//   - Non-periodic: ghost cells appended after real cells, boundary faces
//     point to them. Ghost cell values set by apply_bcs() before each step.
//
// MPI ghost rows: when mpi_mode=true, rows j=0 and j=ny-1 are MPI ghost rows
// (filled by halo exchange). Boundary ghost cells are separate and come after.
// ============================================================================
class Mesh {
public:
    struct Faces {
        std::vector<int_t>   owner;
        std::vector<int_t>   neighbor;
        std::vector<float_t> area;
        std::vector<float_t> normal_x;   // SoA: x-component of normal
        std::vector<float_t> normal_y;   // SoA: y-component of normal
        std::vector<float_t> distance;
        int_t count = 0;

        void resize(int_t n) {
            count = n;
            owner.resize(n);
            neighbor.resize(n);
            area.resize(n);
            normal_x.resize(n);
            normal_y.resize(n);
            distance.resize(n);
        }
    };

private:
    int_t nx, ny;        // ny includes MPI ghost rows if mpi_mode
    int_t real_ny;       // ny without MPI ghost rows
    int_t ncells;        // nx * ny (real + MPI ghost rows)
    int_t ncells_total;  // ncells + boundary ghost cells
    int_t n_ghost_bc;    // number of boundary ghost cells
    float_t hx, hy;
    Float3 v_min, v_max;
    bool mpi_mode_;
    BCSpec bc_specs[4];  // left, right, bottom, top

public:
    std::vector<Float3> centers;
    std::vector<float_t> volumes;

    Faces faces;
    std::vector<int_t> cell_faces;  // cell_faces[cell*4+k] = face index

    // Boundary info: for each face, which boundary it belongs to (-1 = interior)
    // Indexed as Boundary enum: 0=Left, 1=Right, 2=Bottom, 3=Top
    std::vector<int_t> face_boundary_id;

    // For each boundary ghost cell: index of corresponding interior cell
    std::vector<int_t> ghost_interior_map;

    Mesh(int_t nx_, int_t ny_, Float3 min, Float3 max,
         bool mpi_mode = false, const BCSpec bc[4] = nullptr);
    ~Mesh() = default;

    int_t get_nx() const { return nx; }
    int_t get_ny() const { return ny; }
    int_t get_real_ny() const { return real_ny; }
    int_t get_ncells() const { return ncells; }
    int_t get_ncells_total() const { return ncells_total; }
    int_t get_n_ghost_bc() const { return n_ghost_bc; }
    bool is_mpi_mode() const { return mpi_mode_; }
    float_t get_hx() const { return hx; }
    float_t get_hy() const { return hy; }
    Float3 get_vmin() const { return v_min; }
    Float3 get_vmax() const { return v_max; }
    const BCSpec& get_bc(Boundary b) const { return bc_specs[(int)b]; }

    inline int_t cell_index(int_t i, int_t j) const { return j * nx + i; }
    inline int_t idx(int_t i, int_t j) const { return cell_index(i,j); }
    inline int_t face_index(int_t cell, int k) const { return cell * 4 + k; }
};
