#pragma once
#include <string>
#include <cstdio>
#include <cstring>
#include "Mesh.hpp"

class VTKWriter {
public:
    // Byte-swap float для big-endian (VTK legacy binary = big-endian)
    static float swap_float(float val) {
        float result;
        char* src = reinterpret_cast<char*>(&val);
        char* dst = reinterpret_cast<char*>(&result);
        dst[0] = src[3];
        dst[1] = src[2];
        dst[2] = src[1];
        dst[3] = src[0];
        return result;
    }

    static std::string make_vtk_name(int step) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "output_%04d.vtk", step);
        return std::string(buf);
    }

    static void save(Mesh* mesh, int step) {
        std::string filename = make_vtk_name(step);
        FILE* fp = std::fopen(filename.c_str(), "wb");
        if (!fp) return;

        const int_t nx = mesh->get_nx();
        const int_t ny = mesh->get_ny();
        const Float3 vmin = mesh->get_vmin();
        const float_t hx = mesh->get_hx();
        const float_t hy = mesh->get_hy();

        // Header (ASCII)
        std::fprintf(fp, "# vtk DataFile Version 3.0\n");
        std::fprintf(fp, "Heat2D step %d\n", step);
        std::fprintf(fp, "BINARY\n");
        std::fprintf(fp, "DATASET RECTILINEAR_GRID\n");
        std::fprintf(fp, "DIMENSIONS %d %d 1\n", nx, ny);

        // X coordinates (binary, big-endian)
        std::fprintf(fp, "X_COORDINATES %d float\n", nx);
        for (int_t i = 0; i < nx; ++i) {
            float v = swap_float(vmin.x + (i + 0.5f) * hx);
            std::fwrite(&v, sizeof(float), 1, fp);
        }
        std::fprintf(fp, "\n");

        // Y coordinates
        std::fprintf(fp, "Y_COORDINATES %d float\n", ny);
        for (int_t j = 0; j < ny; ++j) {
            float v = swap_float(vmin.y + (j + 0.5f) * hy);
            std::fwrite(&v, sizeof(float), 1, fp);
        }
        std::fprintf(fp, "\n");

        // Z coordinate
        std::fprintf(fp, "Z_COORDINATES 1 float\n");
        float z = swap_float(0.0f);
        std::fwrite(&z, sizeof(float), 1, fp);
        std::fprintf(fp, "\n");

        // Temperature data
        int_t npts = nx * ny;
        std::fprintf(fp, "POINT_DATA %d\n", npts);
        std::fprintf(fp, "SCALARS Temperature float 1\n");
        std::fprintf(fp, "LOOKUP_TABLE default\n");

        float_t* T = mesh->get_T_curr();
        for (int_t c = 0; c < npts; ++c) {
            float v = swap_float(T[c]);
            std::fwrite(&v, sizeof(float), 1, fp);
        }

        std::fclose(fp);
    }

    // Запись из raw-массива T с явными размерами (для MPI gather)
    static void save_raw(const float* T, int_t nx, int_t ny,
                         Float3 vmin, float_t hx, float_t hy, int step) {
        std::string filename = make_vtk_name(step);
        FILE* fp = std::fopen(filename.c_str(), "wb");
        if (!fp) return;

        std::fprintf(fp, "# vtk DataFile Version 3.0\n");
        std::fprintf(fp, "Heat2D step %d\n", step);
        std::fprintf(fp, "BINARY\n");
        std::fprintf(fp, "DATASET RECTILINEAR_GRID\n");
        std::fprintf(fp, "DIMENSIONS %d %d 1\n", nx, ny);

        std::fprintf(fp, "X_COORDINATES %d float\n", nx);
        for (int_t i = 0; i < nx; ++i) {
            float v = swap_float(vmin.x + (i + 0.5f) * hx);
            std::fwrite(&v, sizeof(float), 1, fp);
        }
        std::fprintf(fp, "\n");

        std::fprintf(fp, "Y_COORDINATES %d float\n", ny);
        for (int_t j = 0; j < ny; ++j) {
            float v = swap_float(vmin.y + (j + 0.5f) * hy);
            std::fwrite(&v, sizeof(float), 1, fp);
        }
        std::fprintf(fp, "\n");

        std::fprintf(fp, "Z_COORDINATES 1 float\n");
        float z = swap_float(0.0f);
        std::fwrite(&z, sizeof(float), 1, fp);
        std::fprintf(fp, "\n");

        int_t npts = nx * ny;
        std::fprintf(fp, "POINT_DATA %d\n", npts);
        std::fprintf(fp, "SCALARS Temperature float 1\n");
        std::fprintf(fp, "LOOKUP_TABLE default\n");
        for (int_t c = 0; c < npts; ++c) {
            float v = swap_float(T[c]);
            std::fwrite(&v, sizeof(float), 1, fp);
        }
        std::fclose(fp);
    }

    static void writePVD(int nSteps, const std::string& pvdName = "output.pvd") {
        (void)nSteps; (void)pvdName;
    }
};
