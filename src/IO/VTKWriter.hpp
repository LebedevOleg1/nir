#pragma once
#include "Base/FvmTypes.hpp"
#include "Base/PhysicsType.hpp"
#include "Mesh/Mesh.hpp"
#include <string>
#include <vector>
#include <cstdio>
#include <cmath>

// ============================================================================
// VTKWriter — binary RECTILINEAR_GRID output (big-endian floats).
//
// Format matches the Python animate_kh.py reader.
// ============================================================================
class VTKWriter {
public:
    static std::string make_filename(int step) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "output_%04d.vtk", step);
        return std::string(buf);
    }

    // Write for a single MPI rank
    template<PhysicsType P>
    static void save_fields(const float* U, int ncells_total,
                            Mesh* mesh, int step, float gamma) {
        std::string fn = make_filename(step);
        FILE* fp = std::fopen(fn.c_str(), "wb");
        if (!fp) return;

        int nx = mesh->get_nx();
        int ny = mesh->get_real_ny();
        write_grid(fp, nx, ny, mesh->get_vmin(), mesh->get_hx(), mesh->get_hy());
        std::fprintf(fp, "POINT_DATA %d\n", nx * ny);
        write_physics<P>(fp, U, ncells_total, nx * ny, gamma);
        std::fclose(fp);
    }

    // Write gathered global data (MPI path)
    template<PhysicsType P>
    static void save_raw_fields(const float* U, int ncells_total,
                                int nx, int ny, Vec3 vmin,
                                float hx, float hy, int step, float gamma) {
        std::string fn = make_filename(step);
        FILE* fp = std::fopen(fn.c_str(), "wb");
        if (!fp) return;

        write_grid(fp, nx, ny, vmin, hx, hy);
        std::fprintf(fp, "POINT_DATA %d\n", nx * ny);
        write_physics<P>(fp, U, ncells_total, nx * ny, gamma);
        std::fclose(fp);
    }

private:
    static float swap_float(float v) {
        float r;
        char* s = reinterpret_cast<char*>(&v);
        char* d = reinterpret_cast<char*>(&r);
        d[0]=s[3]; d[1]=s[2]; d[2]=s[1]; d[3]=s[0];
        return r;
    }

    static void write_grid(FILE* fp, int nx, int ny,
                           Vec3 vmin, float hx, float hy) {
        std::fprintf(fp, "# vtk DataFile Version 3.0\nFVM Solver output\nBINARY\n");
        std::fprintf(fp, "DATASET RECTILINEAR_GRID\nDIMENSIONS %d %d 1\n", nx, ny);

        std::fprintf(fp, "X_COORDINATES %d float\n", nx);
        for (int i = 0; i < nx; ++i) {
            float v = swap_float(vmin.x + (i + 0.5f) * hx);
            std::fwrite(&v, sizeof(float), 1, fp);
        }
        std::fprintf(fp, "\nY_COORDINATES %d float\n", ny);
        for (int j = 0; j < ny; ++j) {
            float v = swap_float(vmin.y + (j + 0.5f) * hy);
            std::fwrite(&v, sizeof(float), 1, fp);
        }
        std::fprintf(fp, "\nZ_COORDINATES 1 float\n");
        float z = swap_float(0.0f);
        std::fwrite(&z, sizeof(float), 1, fp);
        std::fprintf(fp, "\n");
    }

    static void write_scalar(FILE* fp, const char* name,
                             const float* data, int npts) {
        std::fprintf(fp, "SCALARS %s float 1\nLOOKUP_TABLE default\n", name);
        for (int c = 0; c < npts; ++c) {
            float v = swap_float(data[c]);
            std::fwrite(&v, sizeof(float), 1, fp);
        }
        std::fprintf(fp, "\n");
    }

    template<PhysicsType P>
    static void write_physics(FILE* fp, const float* U,
                              int ncells_total, int npts, float gamma) {
        if constexpr (P == PhysicsType::Heat) {
            write_scalar(fp, "Temperature", U, npts);
        }
        else if constexpr (P == PhysicsType::Diffusion) {
            write_scalar(fp, "Concentration", U, npts);
        }
        else if constexpr (P == PhysicsType::Euler) {
            write_scalar(fp, "Density", U, npts);

            std::vector<float> vx(npts), vy(npts), pres(npts), mach(npts);
            for (int c = 0; c < npts; ++c) {
                float rho  = U[0 * ncells_total + c];
                float rhou = U[1 * ncells_total + c];
                float rhov = U[2 * ncells_total + c];
                float E    = U[3 * ncells_total + c];
                float u = rhou / rho, v = rhov / rho;
                float p = (gamma - 1.0f) * (E - 0.5f * rho * (u*u + v*v));
                float cs = (gamma * p / rho > 0) ? std::sqrt(gamma * p / rho) : 0.0f;
                vx[c]   = u;
                vy[c]   = v;
                pres[c] = p;
                mach[c] = (cs > 0) ? std::sqrt(u*u + v*v) / cs : 0.0f;
            }
            write_scalar(fp, "VelocityX", vx.data(),   npts);
            write_scalar(fp, "VelocityY", vy.data(),   npts);
            write_scalar(fp, "Pressure",  pres.data(), npts);
            write_scalar(fp, "Mach",      mach.data(), npts);
        }
    }
};
