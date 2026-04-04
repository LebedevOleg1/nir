#pragma once
#include <cstdint>
#include <cuda_runtime.h>

using float_t = float;
using int_t = int;

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

struct Float3 {
    float_t x, y, z;
    HOST_DEVICE Float3(float_t x = 0, float_t y = 0, float_t z = 0) : x(x), y(y), z(z) {}
};