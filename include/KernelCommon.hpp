#pragma once
#include <cmath>

// ============================================================================
// Макросы для кросс-платформенной компиляции CPU/GPU.
// HD — функция компилируется и для хоста, и для девайса.
// DEVICE — только для девайса (GPU).
// FORCE_INLINE — принудительный inline (на GPU — __forceinline__).
// RESTRICT — подсказка компилятору, что указатели не алиасят друг друга,
//            что позволяет более агрессивные оптимизации (vectorization, reordering).
// ============================================================================

#ifdef __CUDACC__
    #define HD __host__ __device__
    #define DEVICE __device__
    #define FORCE_INLINE __forceinline__
#else
    #define HD
    #define DEVICE
    #define FORCE_INLINE inline
#endif

#if defined(__CUDACC__) || defined(__GNUC__) || defined(__clang__)
    #define RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define RESTRICT __restrict
#else
    #define RESTRICT
#endif
