cmake_minimum_required(VERSION 3.10)
project(heat2d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Оптимизация для релиза (по желанию)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# include directory
include_directories(${CMAKE_SOURCE_DIR}/include)

# Все .cpp из src
file(GLOB SOURCES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

add_executable(heat ${SOURCES})

# Для macOS: при желании можно явно указать компилятор (обычно не требуется)
# set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
