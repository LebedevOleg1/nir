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
    faces.resize(4 * ncells);
    cell_faces.resize(4 * ncells);

    for (int_t j = 0; j < ny; ++j) {
        for (int_t i = 0; i < nx; ++i) {
            int_t c = cell_index(i, j);
            centers[c] = Float3(v_min.x + (i + 0.5f) * hx,
                                v_min.y + (j + 0.5f) * hy,
                                0.0f);
            volumes[c] = hx * hy;
        }
    }

    // Для каждой ячейки создаём 4 "локальных" граней:
    // k=0: left, k=1: right, k=2: bottom, k=3: top
    // neighbor с учётом периодичности
    for (int_t j = 0; j < ny; ++j) {
        for (int_t i = 0; i < nx; ++i) {
            int_t c = cell_index(i, j);

            // left face (k=0)
            {
                int k = 0;
                int_t fi = face_index(c, k);
                int_t inb = (i - 1 + nx) % nx;
                int_t jnb = j;
                int_t nb = cell_index(inb, jnb);

                faces[fi].owner = c;
                faces[fi].neighbor = nb;
                faces[fi].area = hy; // вертикальная грань длина hy
                faces[fi].normal = Float3(-1.0f, 0.0f, 0.0f); // внешняя влево
                faces[fi].centroid = Float3(v_min.x + i * hx, v_min.y + (j + 0.5f)*hy, 0.0f);
                faces[fi].distance = std::abs(centers[c].x - centers[nb].x);
                cell_faces[c*4 + k] = fi;
            }

            // right face (k=1)
            {
                int k = 1;
                int_t fi = face_index(c, k);
                int_t inb = (i + 1) % nx;
                int_t jnb = j;
                int_t nb = cell_index(inb, jnb);

                faces[fi].owner = c;
                faces[fi].neighbor = nb;
                faces[fi].area = hy;
                faces[fi].normal = Float3(1.0f, 0.0f, 0.0f);
                faces[fi].centroid = Float3(v_min.x + (i + 1) * hx, v_min.y + (j + 0.5f)*hy, 0.0f);
                faces[fi].distance = std::abs(centers[c].x - centers[nb].x);
                cell_faces[c*4 + k] = fi;
            }

            // bottom face (k=2)
            {
                int k = 2;
                int_t fi = face_index(c, k);
                int_t inb = i;
                int_t jnb = (j - 1 + ny) % ny;
                int_t nb = cell_index(inb, jnb);

                faces[fi].owner = c;
                faces[fi].neighbor = nb;
                faces[fi].area = hx; // горизонтальная грань длина hx
                faces[fi].normal = Float3(0.0f, -1.0f, 0.0f); // вниз
                faces[fi].centroid = Float3(v_min.x + (i + 0.5f)*hx, v_min.y + j * hy, 0.0f);
                faces[fi].distance = std::abs(centers[c].y - centers[nb].y);
                cell_faces[c*4 + k] = fi;
            }

            // top face (k=3)
            {
                int k = 3;
                int_t fi = face_index(c, k);
                int_t inb = i;
                int_t jnb = (j + 1) % ny;
                int_t nb = cell_index(inb, jnb);

                faces[fi].owner = c;
                faces[fi].neighbor = nb;
                faces[fi].area = hx;
                faces[fi].normal = Float3(0.0f, 1.0f, 0.0f); // вверх
                faces[fi].centroid = Float3(v_min.x + (i + 0.5f)*hx, v_min.y + (j + 1) * hy, 0.0f);
                faces[fi].distance = std::abs(centers[c].y - centers[nb].y);
                cell_faces[c*4 + k] = fi;
            }
        }
    }

    data.init(ncells);
}
