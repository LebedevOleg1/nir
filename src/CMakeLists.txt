file(GLOB_RECURSE FVM_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

# solver_cpu_stub.cpp is only for CPU-only builds — exclude it here
list(FILTER FVM_SOURCES EXCLUDE REGEX ".*solver_cpu_stub\\.cpp$")

add_library(fvm_core STATIC ${FVM_SOURCES})

# src/ is the include root: headers referenced as "Base/FvmTypes.hpp" etc.
target_include_directories(fvm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(fvm_core PUBLIC cxx_std_17)

# CUDA: extended lambda support (for ParallelFor lambdas in device code)
# OpenMP passthrough so that solver.cu (which includes solver_impl.inl) gets -fopenmp
target_compile_options(fvm_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)

target_link_libraries(fvm_core PUBLIC
    CUDA::cudart
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)

if(UNIX)
    target_link_libraries(fvm_core PUBLIC m)
endif()
