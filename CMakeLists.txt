cmake_minimum_required(VERSION 3.18)
project(fvm2d_hybrid LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Optional: uncomment and set the right compute capability for your cluster:
# set(CMAKE_CUDA_ARCHITECTURES 80)   # A100
# set(CMAKE_CUDA_ARCHITECTURES 89)   # RTX 4090
# set(CMAKE_CUDA_ARCHITECTURES 70)   # V100

# --- Core library ---
add_subdirectory(src)

# --- Problem executables ---
add_subdirectory(problems)
