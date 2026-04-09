add_executable(sod main.cu)
target_link_libraries(sod PRIVATE fvm_core)
target_include_directories(sod PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
