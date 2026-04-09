add_executable(rt main.cu)
target_link_libraries(rt PRIVATE fvm_core)
target_include_directories(rt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
