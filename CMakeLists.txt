cmake_minimum_required(VERSION 3.30)
project(Pt2Project C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES "native")

enable_language(CUDA)
set_source_files_properties(CUDA/Dijkstra.cu PROPERTIES LANGUAGE CUDA)
set(CUDA_SEPARABLE_COMPILATION ON)


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; "--default-stream per-thread")

include_directories(
        /usr/local/cuda-12.6/include/
)

link_directories(
        /usr/local/cuda-12.6/lib64/
)

include_directories(.)

add_executable(Pt2Project
    BMFord.c
    BMFord.h
    Dijkstra.c
    Dijkstra.h
    GraphParse.c
    GraphParse.h
    GraphSearch.c
    FWarsh.c
    FWarsh.h
    CUDA/Dijkstra.cu
    CUDA/Dijkstra.cuh
    GraphSearch.h
    Main.cpp
        GraphMatrix.cpp
        GraphMatrix.h
        CUDA/FWarsh.cuh
        CUDA/FWarsh.cu)

target_link_libraries(
        Pt2Project
        -lcudadevrt
)
