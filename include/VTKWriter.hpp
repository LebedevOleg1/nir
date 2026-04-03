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
        std::ofstream file(filename);

        const int_t nx = mesh->get_nx();
        const int_t ny = mesh->get_ny();
        const Float3 vmin = mesh->get_vmin();
        const float_t hx = mesh->get_hx();
        const float_t hy = mesh->get_hy();

        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        file << "<RectilinearGrid WholeExtent=\"0 " << nx << " 0 " << ny << " 0 0\">\n";
        file << "<Piece Extent=\"0 " << nx << " 0 " << ny << " 0 0\">\n";

        // PointData: (nx+1)*(ny+1) значений.
        // Ячеечные значения FVM приписываем ближайшему узлу (nearest-neighbor).
        file << "<PointData Scalars=\"Temperature\">\n";
        file << "<DataArray Name=\"Temperature\" type=\"Float32\" format=\"ascii\">\n";
        float_t* T = mesh->get_T_curr();
        int count = 0;
        for (int_t j = 0; j <= ny; ++j) {
            int_t jc = (j < ny) ? j : ny - 1;
            for (int_t i = 0; i <= nx; ++i) {
                int_t ic = (i < nx) ? i : nx - 1;
                write_float(file, T[mesh->idx(ic, jc)]);
                if (++count % 10 == 0) file << "\n";
                else                   file << " ";
            }
        }
        if (count % 10 != 0) file << "\n";
        file << "</DataArray>\n</PointData>\n";

        file << "<Coordinates>\n";
        file << "<DataArray type=\"Float32\" Name=\"X\" format=\"ascii\">\n";
        for (int_t i = 0; i <= nx; ++i) {
            write_float(file, vmin.x + i * hx);
            if ((i + 1) % 10 == 0 || i == nx) file << "\n"; else file << " ";
        }
        file << "</DataArray>\n";
        file << "<DataArray type=\"Float32\" Name=\"Y\" format=\"ascii\">\n";
        for (int_t j = 0; j <= ny; ++j) {
            write_float(file, vmin.y + j * hy);
            if ((j + 1) % 10 == 0 || j == ny) file << "\n"; else file << " ";
        }
        file << "</DataArray>\n";
        file << "<DataArray type=\"Float32\" Name=\"Z\" format=\"ascii\">\n0.0\n</DataArray>\n";
        file << "</Coordinates>\n";

        file << "</Piece>\n</RectilinearGrid>\n</VTKFile>\n";
        file.close();
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
