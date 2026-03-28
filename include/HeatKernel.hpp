#pragma once
#include "KernelCommon.hpp"

// ============================================================================
// Ядро расчёта теплового потока методом конечных объёмов (FVM).
//
// SoA-версия: данные о гранях переданы как отдельные массивы (owner, neighbor,
// area, distance), а не как массив структур GPUFace. Почему это важно:
//
// На GPU потоки одного warp'а (32 потока) выполняются синхронно. Когда
// поток #0 читает face_neighbor[idx], а поток #1 читает face_neighbor[idx+1],
// эти адреса идут подряд → GPU объединяет их в ОДНУ транзакцию памяти
// (coalesced access). С AoS (struct {owner,neighbor,area,dist}) потоки
// читали бы адреса с шагом sizeof(struct) = 16 байт → каждый поток
// тянет свою кэш-линию, эффективность ~25%.
//
// Формула FVM для ячейки i:
//   T_next[i] = T_curr[i] + dt * (sum_k(flux_k) / volume[i] + source[i])
//
// где flux_k = kappa_face * area * (T_neighbor - T_cell) / distance
// — тепловой поток через k-ю грань (закон Фурье в дискретной форме).
// ============================================================================
HD FORCE_INLINE void calculate_heat_flux_core(
    int i,
    const float* RESTRICT T_curr,
    float* RESTRICT T_next,
    const float* RESTRICT volumes,
    const int*   RESTRICT face_owner,      // SoA: владельцы граней
    const int*   RESTRICT face_neighbor,   // SoA: соседи через грань
    const float* RESTRICT face_area,       // SoA: площади граней
    const float* RESTRICT face_distance,   // SoA: расстояния owner↔neighbor
    const int*   RESTRICT cell_faces,      // cell_faces[i*4+k] → индекс грани
    const float* RESTRICT kappa_face,
    const float* RESTRICT source,
    const int ncells,
    const float dt
) {
    if (i >= ncells) return;

    float T_c = T_curr[i];
    float flux_sum = 0.0f;

    // 4 грани на ячейку (2D): left, right, bottom, top
    #ifdef __CUDACC__
    #pragma unroll
    #endif
    for (int k = 0; k < 4; ++k) {
        int face_idx = cell_faces[i * 4 + k];

        // SoA-доступ: каждый массив читается линейно по face_idx
        int neighbor_idx = face_neighbor[face_idx];
        float T_neighbor = T_curr[neighbor_idx];

        // Градиент температуры (центральная разность)
        float grad_T = (T_neighbor - T_c) / face_distance[face_idx];
        // Поток = коэффициент * площадь * градиент (закон Фурье)
        float flux = kappa_face[face_idx] * face_area[face_idx] * grad_T;

        flux_sum += flux;
    }

    // Явная схема Эйлера: T_new = T_old + dt * (дивергенция потока + источник)
    T_next[i] = T_c + dt * (flux_sum / volumes[i] + source[i]);
}
