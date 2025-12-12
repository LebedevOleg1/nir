#pragma once
#include <cmath>

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

struct GPUFace {
    int owner;
    int neighbor;
    float area;
    float distance;
};