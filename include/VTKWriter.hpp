#pragma once
#include <string>
#include <cstdio>
#include <cstring>
#include <cmath>
#include "Mesh.hpp"
#include "PhysicsType.hpp"

class VTKWriter {
public:
    static float swap_float(float val) {
        float result;
        char* src = reinterpret_cast<char*>(&val);
        char* dst = reinterpret_cast<char*>(&result);
        dst[0] = src[3]; dst[1] = src[2]; dst[2] = src[1]; dst[3] = src[0];
        return result;
    }

    static std::string make_vtk_name(int step) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "output_%04d.vtk", step);
        return std::string(buf);
    }

    // Write grid coordinates header
    static void write_grid(FILE* fp, int nx, int ny, Float3 vmin, float hx, float hy) {
        std::fprintf(fp, "# vtk DataFile Version 3.0\n");
        std::fprintf(fp, "FVM Solver output\n");
        std::fprintf(fp, "BINARY\n");
        std::fprintf(fp, "DATASET RECTILINEAR_GRID\n");
        std::fprintf(fp, "DIMENSIONS %d %d 1\n", nx, ny);

        std::fprintf(fp, "X_COORDINATES %d float\n", nx);
        for (int i = 0; i < nx; ++i) {
            float v = swap_float(vmin.x + (i + 0.5f) * hx);
            std::fwrite(&v, sizeof(float), 1, fp);
        }
        std::fprintf(fp, "\n");

        std::fprintf(fp, "Y_COORDINATES %d float\n", ny);
        for (int j = 0; j < ny; ++j) {
            float v = swap_float(vmin.y + (j + 0.5f) * hy);
            std::fwrite(&v, sizeof(float), 1, fp);
        }
        std::fprintf(fp, "\n");

        std::fprintf(fp, "Z_COORDINATES 1 float\n");
        float z = swap_float(0.0f);
        std::fwrite(&z, sizeof(float), 1, fp);
        std::fprintf(fp, "\n");
    }

    static void write_scalar(FILE* fp, const char* name, const float* data,
                              int stride, int npts) {
        std::fprintf(fp, "SCALARS %s float 1\n", name);
        std::fprintf(fp, "LOOKUP_TABLE default\n");
        for (int c = 0; c < npts; ++c) {
            float v = swap_float(data[c]);
            std::fwrite(&v, sizeof(float), 1, fp);
        }
    }

    // --- Save from Mesh (single rank) ---
    template<PhysicsType P>
    static void save_fields(const float* U, int ncells_total,
                            Mesh* mesh, int step, float gamma) {
        std::string filename = make_vtk_name(step);
        FILE* fp = std::fopen(filename.c_str(), "wb");
        if (!fp) return;

        int nx = mesh->get_nx();
        int ny = mesh->get_ny();
        int npts = nx * ny;

        write_grid(fp, nx, ny, mesh->get_vmin(), mesh->get_hx(), mesh->get_hy());
        std::fprintf(fp, "POINT_DATA %d\n", npts);

        write_physics_fields<P>(fp, U, ncells_total, npts, gamma);
        std::fclose(fp);
    }

    // --- Save from raw array (MPI gather) ---
    template<PhysicsType P>
    static void save_raw_fields(const float* U, int ncells_total,
                                 int nx, int ny, Float3 vmin,
                                 float hx, float hy, int step, float gamma) {
        std::string filename = make_vtk_name(step);
        FILE* fp = std::fopen(filename.c_str(), "wb");
        if (!fp) return;

        int npts = nx * ny;
        write_grid(fp, nx, ny, vmin, hx, hy);
        std::fprintf(fp, "POINT_DATA %d\n", npts);

        write_physics_fields<P>(fp, U, ncells_total, npts, gamma);
        std::fclose(fp);
    }

private:
    template<PhysicsType P>
    static void write_physics_fields(FILE* fp, const float* U, int ncells_total,
                                      int npts, float gamma) {
        constexpr int NVAR = PhysicsTraits<P>::NVAR;

        if constexpr (P == PhysicsType::Heat) {
            write_scalar(fp, "Temperature", U, ncells_total, npts);
        }
        else if constexpr (P == PhysicsType::Diffusion) {
            write_scalar(fp, "Concentration", U, ncells_total, npts);
        }
        else if constexpr (P == PhysicsType::Euler) {
            // Density
            write_scalar(fp, "Density", U, ncells_total, npts);

            // Compute and write derived fields: velocity, pressure, Mach
            std::vector<float> vel_x(npts), vel_y(npts), pressure(npts), mach(npts);
            for (int c = 0; c < npts; ++c) {
                float rho  = U[0 * ncells_total + c];
                float rhou = U[1 * ncells_total + c];
                float rhov = U[2 * ncells_total + c];
                float E    = U[3 * ncells_total + c];
                float u = rhou / rho;
                float v = rhov / rho;
                float p = (gamma - 1.0f) * (E - 0.5f * rho * (u*u + v*v));
                float c_s = (gamma * p / rho > 0) ? std::sqrt(gamma * p / rho) : 0.0f;
                float speed = std::sqrt(u*u + v*v);

                vel_x[c] = u;
                vel_y[c] = v;
                pressure[c] = p;
                mach[c] = (c_s > 0) ? speed / c_s : 0.0f;
            }

            write_scalar(fp, "VelocityX", vel_x.data(), npts, npts);
            write_scalar(fp, "VelocityY", vel_y.data(), npts, npts);
            write_scalar(fp, "Pressure", pressure.data(), npts, npts);
            write_scalar(fp, "Mach", mach.data(), npts, npts);
        }
    }
};
