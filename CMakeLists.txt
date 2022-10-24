cmake_minimum_required(VERSION 3.23)
project(assignment2 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(assignment2 main.cu FileHandler.cu FileHandler.cuh Image.cu Image.cuh)

set_target_properties(assignment2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
