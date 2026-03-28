cmake_minimum_required(VERSION 3.18)

# Проект использует три языка/технологии:
#   CXX  — C++17, основной язык
#   CUDA — NVIDIA GPU compute (nvcc компилирует .cu файлы)
project(heat2d_hybrid LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Зависимости ---

# CUDA Toolkit: предоставляет CUDA::cudart, thrust и пр.
find_package(CUDAToolkit REQUIRED)

# MPI: предоставляет MPI::MPI_CXX для межпроцессного обмена
find_package(MPI REQUIRED)

# --- Настройки компиляции ---

# Архитектура GPU закомментирована — CMake определит автоматически,
# либо на кластере можно задать через -DCMAKE_CUDA_ARCHITECTURES=XX
# set(CMAKE_CUDA_ARCHITECTURES 89)

# Подключаем заголовки
include_directories(${CMAKE_SOURCE_DIR}/include)

# --- Сборка ---

# Собираем все .cpp и .cu из src/
file(GLOB SOURCES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.cu"
)

add_executable(heat ${SOURCES})

# --- Линковка ---

# CUDA Runtime + device runtime (для dynamic parallelism если понадобится)
target_link_libraries(heat PRIVATE CUDA::cudart)
target_link_libraries(heat ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY})

# MPI
target_link_libraries(heat PRIVATE MPI::MPI_CXX)

# Math library (Linux)
if(UNIX)
    target_link_libraries(heat PRIVATE m)
endif()
