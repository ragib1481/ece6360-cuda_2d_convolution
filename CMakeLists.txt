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

configure_file(hereford.tga
        hereford.tga COPYONLY)

configure_file(M104_ngc4594_sombrero_galaxy.tga
        M104_ngc4594_sombrero_galaxy.tga COPYONLY)

configure_file(whirlpool.tga
        whirlpool.tga COPYONLY)

add_executable(assignment2 main.cu)

set_target_properties(assignment2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
