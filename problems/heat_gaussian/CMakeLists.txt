add_executable(heat main.cu)
target_link_libraries(heat PRIVATE fvm_core)
target_include_directories(heat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
