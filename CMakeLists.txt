cmake_minimum_required(VERSION 3.10)
project(heat2d_opencl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB SOURCES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

find_package(OpenCL)
if (OpenCL_FOUND)
  message(STATUS "Found OpenCL")
  include_directories(${OpenCL_INCLUDE_DIRS})
  add_executable(heat ${SOURCES})
  if(TARGET OpenCL::OpenCL)
    target_link_libraries(heat PRIVATE OpenCL::OpenCL)
  else()
    target_link_libraries(heat PRIVATE ${OpenCL_LIBRARIES})
  endif()
else()
  message(WARNING "OpenCL not found — target will build, but will run CPU fallback.")
  add_executable(heat ${SOURCES})
endif()
