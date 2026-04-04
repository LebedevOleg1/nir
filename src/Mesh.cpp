#include "Mesh.hpp"
#include <cmath>

Mesh::Mesh(int_t nx_, int_t ny_, Float3 min, Float3 max, bool mpi_mode)
    : nx(nx_), real_ny(ny_), v_min(min), v_max(max), mpi_mode_(mpi_mode)
{
    // В MPI-режиме добавляем 2 ghost-строки (сверху и снизу)
    ny = mpi_mode ? (real_ny + 2) : real_ny;
    ncells = nx * ny;

    // hy вычисляется по РЕАЛЬНЫМ строкам (без ghost)
    hx = (v_max.x - v_min.x) / static_cast<float_t>(nx);
    hy = (v_max.y - v_min.y) / static_cast<float_t>(real_ny);

    centers.resize(ncells);
    volumes.resize(ncells);
    faces.resize(4 * ncells);
    cell_faces.resize(4 * ncells);

    kappa.resize(ncells, 1.0f);
    kappa_face.resize(4 * ncells);
    source.resize(ncells, 0.0f);

    // --- Центры и объёмы ячеек ---
    for (int_t j = 0; j < ny; ++j) {
        for (int_t i = 0; i < nx; ++i) {
            int_t c = cell_index(i, j);
            float_t cy;
            if (mpi_mode) {
                // j=0: ghost снизу, j=1..real_ny: реальные, j=real_ny+1: ghost сверху
                cy = v_min.y + (j - 1 + 0.5f) * hy;
            } else {
                cy = v_min.y + (j + 0.5f) * hy;
            }
            centers[c] = Float3(v_min.x + (i + 0.5f) * hx, cy, 0.0f);
            volumes[c] = hx * hy;
        }
    }

    // --- Построение граней ---
    for (int_t j = 0; j < ny; ++j) {
        for (int_t i = 0; i < nx; ++i) {
            int_t c = cell_index(i, j);

            // left (k=0) — X-периодика всегда
            {
                int k = 0;
                int_t fi = face_index(c, k);
                int_t nb = cell_index((i - 1 + nx) % nx, j);
                faces.owner[fi]    = c;
                faces.neighbor[fi] = nb;
                faces.area[fi]     = hy;
                faces.normal[fi]   = Float3(-1.0f, 0.0f, 0.0f);
                faces.centroid[fi] = Float3(v_min.x + i * hx, centers[c].y, 0.0f);
                faces.distance[fi] = hx;
                cell_faces[c*4 + k] = fi;
            }

            // right (k=1)
            {
                int k = 1;
                int_t fi = face_index(c, k);
                int_t nb = cell_index((i + 1) % nx, j);
                faces.owner[fi]    = c;
                faces.neighbor[fi] = nb;
                faces.area[fi]     = hy;
                faces.normal[fi]   = Float3(1.0f, 0.0f, 0.0f);
                faces.centroid[fi] = Float3(v_min.x + (i + 1) * hx, centers[c].y, 0.0f);
                faces.distance[fi] = hx;
                cell_faces[c*4 + k] = fi;
            }

            // bottom (k=2)
            {
                int k = 2;
                int_t fi = face_index(c, k);
                int_t nb;
                if (mpi_mode) {
                    // Ghost строка j=0: сосед — сама себя (данные перезаписываются halo exchange)
                    // Реальная строка j=1: сосед — ghost j=0
                    // Остальные: j-1
                    nb = (j == 0) ? c : cell_index(i, j - 1);
                } else {
                    nb = cell_index(i, (j - 1 + ny) % ny);
                }
                faces.owner[fi]    = c;
                faces.neighbor[fi] = nb;
                faces.area[fi]     = hx;
                faces.normal[fi]   = Float3(0.0f, -1.0f, 0.0f);
                faces.centroid[fi] = Float3(centers[c].x, centers[c].y - 0.5f*hy, 0.0f);
                faces.distance[fi] = hy;
                cell_faces[c*4 + k] = fi;
            }

            // top (k=3)
            {
                int k = 3;
                int_t fi = face_index(c, k);
                int_t nb;
                if (mpi_mode) {
                    // Ghost строка j=ny-1: сосед — сама себя
                    // Реальная строка j=real_ny: сосед — ghost j=ny-1
                    // Остальные: j+1
                    nb = (j == ny - 1) ? c : cell_index(i, j + 1);
                } else {
                    nb = cell_index(i, (j + 1) % ny);
                }
                faces.owner[fi]    = c;
                faces.neighbor[fi] = nb;
                faces.area[fi]     = hx;
                faces.normal[fi]   = Float3(0.0f, 1.0f, 0.0f);
                faces.centroid[fi] = Float3(centers[c].x, centers[c].y + 0.5f*hy, 0.0f);
                faces.distance[fi] = hy;
                cell_faces[c*4 + k] = fi;
            }
        }
    }

    // --- Теплопроводность на гранях (гармоническое среднее) ---
    for (int_t fi = 0; fi < 4 * ncells; ++fi) {
        int_t o = faces.owner[fi];
        int_t n = faces.neighbor[fi];
        float_t d  = faces.distance[fi];
        float_t d1 = d / 2.0f;
        float_t d2 = d / 2.0f;
        kappa_face[fi] = (d1 + d2) / (d1/kappa[o] + d2/kappa[n]);
    }

    data.init(ncells);
}
