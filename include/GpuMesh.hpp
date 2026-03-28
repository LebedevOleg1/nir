#pragma once

#include "Mesh.hpp"
#include <thrust/device_vector.h>

// ============================================================================
// GpuMesh — «GPU-близнец» класса Mesh.
//
// Хранит те же данные (грани, объёмы, температуру и т.д.), но в видеопамяти
// GPU, используя thrust::device_vector. Это обёртка над cudaMalloc/cudaFree,
// которая:
//   1) Автоматически выделяет/освобождает GPU-память (RAII)
//   2) При присваивании device_vector = std::vector делает cudaMemcpy H→D
//   3) При thrust::copy(device → host) делает cudaMemcpy D→H
//
// Грани — SoA (отдельные массивы owner, neighbor, area, distance),
// что даёт coalesced memory access на GPU.
//
// Для передачи в CUDA-ядра используем raw-указатели:
//   thrust::raw_pointer_cast(vec.data())
// потому что ядра (__global__ функции) не могут работать с thrust-объектами.
// ============================================================================

// SoA-хранилище граней на GPU
struct GpuFaces {
    thrust::device_vector<int>   owner;
    thrust::device_vector<int>   neighbor;
    thrust::device_vector<float> area;
    thrust::device_vector<float> distance;
};

struct GpuMesh {
    // Двойной буфер температуры (ping-pong): на чётном шаге пишем curr→next,
    // на нечётном — next→curr. swap_buffers() меняет их местами.
    thrust::device_vector<float> T_curr;
    thrust::device_vector<float> T_next;

    thrust::device_vector<float> volumes;
    thrust::device_vector<float> kappa_face;
    thrust::device_vector<float> source;
    thrust::device_vector<int>   cell_faces;

    GpuFaces faces;

    int ncells = 0;
    int nfaces = 0;

    // Загрузить все данные из CPU-сетки в GPU-память.
    // Вызывается один раз при инициализации.
    void upload(const Mesh& mesh);

    // Скопировать текущую температуру T_curr обратно в CPU-сетку.
    // Нужно для VTK-вывода и halo-обмена через MPI.
    void download_T(Mesh& mesh);

    // Загрузить обновлённый источниковый член в GPU.
    void upload_source(const std::vector<float>& src);

    // Поменять местами T_curr и T_next после шага по времени.
    void swap_buffers();
};
