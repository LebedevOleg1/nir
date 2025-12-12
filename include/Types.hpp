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