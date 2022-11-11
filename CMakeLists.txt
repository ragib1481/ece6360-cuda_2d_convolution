cmake_minimum_required(VERSION 3.23)
project(assignment2 CUDA)

set(CMAKE_CUDA_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS "-fmad=false")
configure_file(bansberia.tga
        bansberia.tga COPYONLY)
add_executable(assignment2 main.cu FileHandler.cu FileHandler.cuh Image.cu Image.cuh)

set_target_properties(assignment2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
