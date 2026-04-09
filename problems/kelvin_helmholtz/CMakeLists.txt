add_executable(kh main.cu)
target_link_libraries(kh PRIVATE fvm_core)
target_include_directories(kh PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
