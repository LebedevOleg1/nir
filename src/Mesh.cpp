#include "Mesh.hpp"
#include <cmath>

Mesh::Mesh(int_t nx_, int_t ny_, Float3 min, Float3 max)
    : nx(nx_), ny(ny_), v_min(min), v_max(max)
{
    ncells = nx * ny;
    hx = (v_max.x - v_min.x) / static_cast<float_t>(nx);
    hy = (v_max.y - v_min.y) / static_cast<float_t>(ny);

    centers.resize(ncells);
    volumes.resize(ncells);

    // SoA: выделяем 4*ncells граней (left, right, bottom, top для каждой ячейки)
    faces.resize(4 * ncells);
    cell_faces.resize(4 * ncells);

    kappa.resize(ncells, 1.0f);
    kappa_face.resize(4 * ncells);
    source.resize(ncells, 0.0f);

    // --- Вычисление центров и объёмов ячеек ---
    for (int_t j = 0; j < ny; ++j) {
        for (int_t i = 0; i < nx; ++i) {
            int_t c = cell_index(i, j);
            centers[c] = Float3(v_min.x + (i + 0.5f) * hx,
                                v_min.y + (j + 0.5f) * hy,
                                0.0f);
            volumes[c] = hx * hy;
        }
    }

    // --- Построение граней (SoA) ---
    // Каждая ячейка имеет 4 грани: k=0 left, k=1 right, k=2 bottom, k=3 top.
    // Периодические граничные условия: сетка «замкнута» в тор.
    for (int_t j = 0; j < ny; ++j) {
        for (int_t i = 0; i < nx; ++i) {
            int_t c = cell_index(i, j);

            // left (k=0)
            {
                int k = 0;
                int_t fi = face_index(c, k);
                int_t nb = cell_index((i - 1 + nx) % nx, j);

                faces.owner[fi]    = c;
                faces.neighbor[fi] = nb;
                faces.area[fi]     = hy;
                faces.normal[fi]   = Float3(-1.0f, 0.0f, 0.0f);
                faces.centroid[fi] = Float3(v_min.x + i * hx, v_min.y + (j + 0.5f)*hy, 0.0f);
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
                faces.centroid[fi] = Float3(v_min.x + (i + 1) * hx, v_min.y + (j + 0.5f)*hy, 0.0f);
                faces.distance[fi] = hx;
                cell_faces[c*4 + k] = fi;
            }

            // bottom (k=2)
            {
                int k = 2;
                int_t fi = face_index(c, k);
                int_t nb = cell_index(i, (j - 1 + ny) % ny);

                faces.owner[fi]    = c;
                faces.neighbor[fi] = nb;
                faces.area[fi]     = hx;
                faces.normal[fi]   = Float3(0.0f, -1.0f, 0.0f);
                faces.centroid[fi] = Float3(v_min.x + (i + 0.5f)*hx, v_min.y + j * hy, 0.0f);
                faces.distance[fi] = hy;
                cell_faces[c*4 + k] = fi;
            }

            // top (k=3)
            {
                int k = 3;
                int_t fi = face_index(c, k);
                int_t nb = cell_index(i, (j + 1) % ny);

                faces.owner[fi]    = c;
                faces.neighbor[fi] = nb;
                faces.area[fi]     = hx;
                faces.normal[fi]   = Float3(0.0f, 1.0f, 0.0f);
                faces.centroid[fi] = Float3(v_min.x + (i + 0.5f)*hx, v_min.y + (j + 1) * hy, 0.0f);
                faces.distance[fi] = hy;
                cell_faces[c*4 + k] = fi;
            }
        }
    }

    // --- Теплопроводность на гранях (гармоническое среднее) ---
    // Гармоническое среднее правильно учитывает скачок теплопроводности
    // на границе двух материалов (аналогия: последовательное сопротивление).
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
