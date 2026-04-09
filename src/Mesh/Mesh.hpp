#pragma once
#include "Base/FvmTypes.hpp"
#include "Base/FvmMacros.hpp"
#include "Mesh/BC.hpp"
#include <vector>

// ============================================================================
// Mesh — 2D rectangular FVM mesh (geometry only, no physics).
//
// Faces stored as SoA for coalesced GPU access.
// 4 faces per cell: left(0), right(1), bottom(2), top(3).
//
// Boundary ghost cells:
//   Non-periodic boundaries get ghost cells appended after real cells.
//   Ghost values are set by apply_bcs() before each time step.
//
// MPI ghost rows:
//   When mpi_mode=true, rows j=0 and j=ny-1 are halo rows for MPI exchange.
//   They are separate from BC ghost cells.
// ============================================================================
class Mesh {
public:
    struct Faces {
        std::vector<Int>  owner;
        std::vector<Int>  neighbor;
        std::vector<Real> area;
        std::vector<Real> normal_x;
        std::vector<Real> normal_y;
        std::vector<Real> distance;
        Int count = 0;

        void resize(Int n) {
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
    Int  nx, ny;           // ny includes MPI ghost rows if mpi_mode
    Int  real_ny;          // ny without MPI ghost rows
    Int  ncells;           // nx * ny
    Int  ncells_total;     // ncells + BC ghost cells
    Int  n_ghost_bc;       // number of BC ghost cells
    Real hx, hy;
    Vec3 v_min, v_max;
    bool mpi_mode_;
    BCSpec bc_specs[4];    // left, right, bottom, top

public:
    std::vector<Vec3> centers;
    std::vector<Real> volumes;

    Faces faces;
    std::vector<Int> cell_faces;       // cell_faces[cell*4+k] = face index
    std::vector<Int> face_boundary_id; // -1 = interior, else Boundary enum value
    std::vector<Int> ghost_interior_map;

    Mesh(Int nx_, Int ny_, Vec3 vmin, Vec3 vmax,
         bool mpi_mode = false, const BCSpec bc[4] = nullptr);
    ~Mesh() = default;

    Int  get_nx()           const { return nx; }
    Int  get_ny()           const { return ny; }
    Int  get_real_ny()      const { return real_ny; }
    Int  get_ncells()       const { return ncells; }
    Int  get_ncells_total() const { return ncells_total; }
    Int  get_n_ghost_bc()   const { return n_ghost_bc; }
    bool is_mpi_mode()      const { return mpi_mode_; }
    Real get_hx()           const { return hx; }
    Real get_hy()           const { return hy; }
    Vec3 get_vmin()         const { return v_min; }
    Vec3 get_vmax()         const { return v_max; }
    const BCSpec& get_bc(Boundary b) const { return bc_specs[(int)b]; }

    FVM_INLINE Int cell_index(Int i, Int j)       const { return j * nx + i; }
    FVM_INLINE Int idx(Int i, Int j)               const { return cell_index(i, j); }
    FVM_INLINE Int face_index(Int cell, int k)     const { return cell * 4 + k; }
};
