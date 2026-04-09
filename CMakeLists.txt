cmake_minimum_required(VERSION 3.18)
project(fvm2d_hybrid LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Set compute capability for the target cluster (V100 = SM 70).
# Override at configure time if needed: cmake -DCMAKE_CUDA_ARCHITECTURES=80 ...
set(CMAKE_CUDA_ARCHITECTURES 70)   # V100

# --- Core library ---
add_subdirectory(src)

# --- Problem executables ---
add_subdirectory(problems)
