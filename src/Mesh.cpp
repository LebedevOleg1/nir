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

    kappa.resize(ncells, 1.0f);        
    kappa_face.resize(4 * ncells);
    source.resize(ncells, 0.0f);       

    // Сначала инициализируем центры и объемы
    for (int_t j = 0; j < ny; ++j) {
        for (int_t i = 0; i < nx; ++i) {
            int_t c = cell_index(i, j);
            centers[c] = Float3(v_min.x + (i + 0.5f) * hx,
                                v_min.y + (j + 0.5f) * hy,
                                0.0f);
            volumes[c] = hx * hy;
        }
    }

    // Затем инициализируем грани
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
                faces[fi].area = hy;
                faces[fi].normal = Float3(-1.0f, 0.0f, 0.0f);
                faces[fi].centroid = Float3(v_min.x + i * hx, v_min.y + (j + 0.5f)*hy, 0.0f);
                faces[fi].distance = hx; // фиксированное расстояние для регулярной сетки
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
                faces[fi].distance = hx;
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
                faces[fi].area = hx;
                faces[fi].normal = Float3(0.0f, -1.0f, 0.0f);
                faces[fi].centroid = Float3(v_min.x + (i + 0.5f)*hx, v_min.y + j * hy, 0.0f);
                faces[fi].distance = hy;
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
                faces[fi].normal = Float3(0.0f, 1.0f, 0.0f);
                faces[fi].centroid = Float3(v_min.x + (i + 0.5f)*hx, v_min.y + (j + 1) * hy, 0.0f);
                faces[fi].distance = hy;
                cell_faces[c*4 + k] = fi;
            }
        }
    }

    for (int_t fi = 0; fi < 4 * ncells; ++fi) {
        int_t owner = faces[fi].owner;
        int_t neighbor = faces[fi].neighbor;
        
        float_t d = faces[fi].distance;
        float_t d1 = d / 2.0f;
        float_t d2 = d / 2.0f;
        kappa_face[fi] = (d1 + d2) / (d1/kappa[owner] + d2/kappa[neighbor]);
    }

    data.init(ncells);
}
