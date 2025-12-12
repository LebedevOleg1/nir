cmake_minimum_required(VERSION 3.18)

# Включаем поддержку и C++, и CUDA
project(heat2d_hybrid LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Находим инструменты CUDA
find_package(CUDAToolkit REQUIRED)

# --- НАСТРОЙКИ КОМПИЛЯЦИИ ---

# 1. Архитектура RTX 4050 (Ada Lovelace)
set(CMAKE_CUDA_ARCHITECTURES 89)

# 2. Флаг, чтобы CUDA не ругалась на новую версию Visual Studio
# Также можно добавить -O3 для оптимизации, но лучше переключать режим сборки в VS Code
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

# Подключаем папку include
include_directories(${CMAKE_SOURCE_DIR}/include)

# --- СБОРКА ИСХОДНИКОВ ---

# Автоматически находим все .cpp и .cu файлы в папке src
file(GLOB SOURCES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.cu"
)

# Создаем исполняемый файл
add_executable(heat ${SOURCES})

# --- ЛИНКОВКА ---

# Линкуем стандартную библиотеку CUDA Runtime
target_link_libraries(heat PRIVATE CUDA::cudart)

# Линкуем математическую библиотеку (для sin/cos на CPU, если нужно)
if(UNIX)
    target_link_libraries(heat PRIVATE m)
endif()