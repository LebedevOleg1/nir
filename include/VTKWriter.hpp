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

    static void save(Mesh* mesh, int step) {
        std::string filename = make_vtr_name(step);
        std::ofstream file(filename);
        file << std::setprecision(6);

        const int_t nx = mesh->get_nx();
        const int_t ny = mesh->get_ny();
        const Float3 vmin = mesh->get_vmin();
        const float_t hx = mesh->get_hx();
        const float_t hy = mesh->get_hy();

        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        // Extent: nx*ny CELLS, nodes 0..nx in X, 0..ny in Y, Z=0 (2D degenerate)
        file << "<RectilinearGrid WholeExtent=\"0 " << nx << " 0 " << ny << " 0 0\">\n";
        file << "<Piece Extent=\"0 " << nx << " 0 " << ny << " 0 0\">\n";

        // CellData: one value per cell (cell-centered FVM data)
        file << "<CellData Scalars=\"Temperature\">\n";
        file << "<DataArray Name=\"Temperature\" type=\"Float32\" format=\"ascii\">\n";
        float_t* T = mesh->get_T_curr();
        for (int_t j = 0; j < ny; ++j) {
            for (int_t i = 0; i < nx; ++i) {
                file << T[mesh->idx(i,j)] << " ";
            }
            file << "\n";
        }
        file << "</DataArray>\n</CellData>\n";

        file << "<Coordinates>\n";
        // X node coords: nx+1 cell-boundary positions
        file << "<DataArray type=\"Float32\" Name=\"X\">\n";
        for (int_t i = 0; i <= nx; ++i) file << vmin.x + i*hx << " ";
        file << "\n</DataArray>\n";
        // Y node coords: ny+1 cell-boundary positions
        file << "<DataArray type=\"Float32\" Name=\"Y\">\n";
        for (int_t j = 0; j <= ny; ++j) file << vmin.y + j*hy << " ";
        file << "\n</DataArray>\n";
        // Z: single node for 2D
        file << "<DataArray type=\"Float32\" Name=\"Z\">\n0.0\n</DataArray>\n";
        file << "</Coordinates>\n";

        file << "</Piece>\n</RectilinearGrid>\n</VTKFile>";
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