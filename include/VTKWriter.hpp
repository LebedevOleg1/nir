#pragma once
#include <string>
#include <fstream>
#include <iomanip>
#include <cstdio>
#include "Mesh.hpp"

class VTKWriter {
public:
    static std::string make_vtr_name(int step) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "output_%04d.vtr", step);
        return std::string(buf);
    }

    // Вспомогательная функция: пишет float через snprintf (игнорирует локаль,
    // всегда использует точку как десятичный разделитель).
    static void write_float(std::ofstream& file, float val) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.6g", val);
        file << buf;
    }

    static void save(Mesh* mesh, int step) {
        std::string filename = make_vtr_name(step);
        FILE* fp = std::fopen(filename.c_str(), "w");
        if (!fp) return;

        const int_t nx = mesh->get_nx();
        const int_t ny = mesh->get_ny();
        const Float3 vmin = mesh->get_vmin();
        const float_t hx = mesh->get_hx();
        const float_t hy = mesh->get_hy();
        const int_t ncells = nx * ny;

        std::fprintf(fp, "<?xml version=\"1.0\"?>\n");
        std::fprintf(fp, "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
        std::fprintf(fp, "<RectilinearGrid WholeExtent=\"0 %d 0 %d 0 0\">\n", nx, ny);
        std::fprintf(fp, "<Piece Extent=\"0 %d 0 %d 0 0\">\n", nx, ny);

        // CellData: nx*ny значений — естественное представление для FVM
        std::fprintf(fp, "<CellData Scalars=\"Temperature\">\n");
        std::fprintf(fp, "<DataArray type=\"Float32\" Name=\"Temperature\" "
                         "NumberOfTuples=\"%d\" format=\"ascii\">\n", ncells);
        float_t* T = mesh->get_T_curr();
        for (int_t c = 0; c < ncells; ++c) {
            std::fprintf(fp, "%.6g", T[c]);
            if ((c + 1) % 10 == 0) std::fprintf(fp, "\n");
            else                    std::fprintf(fp, " ");
        }
        if (ncells % 10 != 0) std::fprintf(fp, "\n");
        std::fprintf(fp, "</DataArray>\n</CellData>\n");

        // Координаты узлов: nx+1 по X, ny+1 по Y, 1 по Z
        std::fprintf(fp, "<Coordinates>\n");

        std::fprintf(fp, "<DataArray type=\"Float32\" Name=\"X\" "
                         "NumberOfTuples=\"%d\" format=\"ascii\">\n", nx + 1);
        for (int_t i = 0; i <= nx; ++i) {
            std::fprintf(fp, "%.6g", (double)(vmin.x + i * hx));
            if ((i + 1) % 10 == 0 || i == nx) std::fprintf(fp, "\n");
            else                               std::fprintf(fp, " ");
        }
        std::fprintf(fp, "</DataArray>\n");

        std::fprintf(fp, "<DataArray type=\"Float32\" Name=\"Y\" "
                         "NumberOfTuples=\"%d\" format=\"ascii\">\n", ny + 1);
        for (int_t j = 0; j <= ny; ++j) {
            std::fprintf(fp, "%.6g", (double)(vmin.y + j * hy));
            if ((j + 1) % 10 == 0 || j == ny) std::fprintf(fp, "\n");
            else                               std::fprintf(fp, " ");
        }
        std::fprintf(fp, "</DataArray>\n");

        std::fprintf(fp, "<DataArray type=\"Float32\" Name=\"Z\" "
                         "NumberOfTuples=\"1\" format=\"ascii\">\n0.0\n</DataArray>\n");
        std::fprintf(fp, "</Coordinates>\n");

        std::fprintf(fp, "</Piece>\n</RectilinearGrid>\n</VTKFile>\n");
        std::fclose(fp);
    }

    static void writePVD(int nSteps, const std::string& pvdName = "output.pvd") {
        std::ofstream pvd(pvdName);
        pvd << "<?xml version=\"1.0\"?>\n";
        pvd << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        pvd << "<Collection>\n";
        for (int s = 0; s < nSteps; ++s) {
            std::string fname = make_vtr_name(s);
            pvd << "<DataSet timestep=\"" << s << "\" group=\"\" part=\"0\" file=\"" << fname << "\"/>\n";
        }
        pvd << "</Collection>\n</VTKFile>\n";
        pvd.close();
    }
};
