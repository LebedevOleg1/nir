add_executable(vortex main.cu)
target_link_libraries(vortex PRIVATE fvm_core)
target_include_directories(vortex PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
