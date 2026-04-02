#pragma once
#include <string>
#include <fstream>
#include <iomanip>
#include <cstdio>
#include "Mesh.hpp"

class VTKWriter {
public:
    static std::string make_vtk_name(int step) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "output_%04d.vtk", step);
        return std::string(buf);
    }

    static void save(Mesh* mesh, int step) {
        std::string filename = make_vtk_name(step);
        std::ofstream file(filename);
        file.imbue(std::locale("C"));
        file << std::setprecision(6);

        const int_t nx = mesh->get_nx();
        const int_t ny = mesh->get_ny();
        const Float3 vmin = mesh->get_vmin();
        const float_t hx = mesh->get_hx();
        const float_t hy = mesh->get_hy();

        // Legacy VTK ASCII format — поддерживается всеми версиями ParaView
        file << "# vtk DataFile Version 3.0\n";
        file << "Heat diffusion step " << step << "\n";
        file << "ASCII\n";
        file << "DATASET RECTILINEAR_GRID\n";
        file << "DIMENSIONS " << (nx + 1) << " " << (ny + 1) << " 1\n";

        // Узловые координаты: nx+1 значений по X
        file << "X_COORDINATES " << (nx + 1) << " float\n";
        for (int_t i = 0; i <= nx; ++i)
            file << vmin.x + i * hx << " ";
        file << "\n";

        // Узловые координаты: ny+1 значений по Y
        file << "Y_COORDINATES " << (ny + 1) << " float\n";
        for (int_t j = 0; j <= ny; ++j)
            file << vmin.y + j * hy << " ";
        file << "\n";

        file << "Z_COORDINATES 1 float\n0\n";

        // Ячеечные данные: nx*ny значений (FVM cell-centered)
        file << "CELL_DATA " << (nx * ny) << "\n";
        file << "SCALARS Temperature float 1\n";
        file << "LOOKUP_TABLE default\n";

        float_t* T = mesh->get_T_curr();
        for (int_t j = 0; j < ny; ++j) {
            for (int_t i = 0; i < nx; ++i)
                file << T[mesh->idx(i, j)] << " ";
            file << "\n";
        }

        file.close();
    }

    static void writePVD(int nSteps, const std::string& pvdName = "output.pvd") {
        std::ofstream pvd(pvdName);
        pvd << "<?xml version=\"1.0\"?>\n";
        pvd << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        pvd << "<Collection>\n";
        for (int s = 0; s < nSteps; ++s) {
            std::string fname = make_vtk_name(s);
            pvd << "<DataSet timestep=\"" << s << "\" group=\"\" part=\"0\" file=\"" << fname << "\"/>\n";
        }
        pvd << "</Collection>\n</VTKFile>\n";
        pvd.close();
    }
};
