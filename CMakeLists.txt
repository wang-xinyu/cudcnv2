cmake_minimum_required(VERSION 2.6)

project(dcnv2)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/cuda-9.0/targets/aarch64-linux/include)
link_directories(/usr/local/cuda-9.0/targets/aarch64-linux/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(im2col ${PROJECT_SOURCE_DIR}/cuda/dcn_v2_im2col_cuda.cu)

add_executable(app ${PROJECT_SOURCE_DIR}/dcn_v2_cuda.c ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(app cudart cublas)
target_link_libraries(app im2col)

add_definitions(-O2 -pthread)

