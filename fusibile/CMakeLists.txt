cmake_minimum_required (VERSION 2.8)
project (fusibile)

find_package(OpenCV REQUIRED )
find_package(CUDA 6.0 REQUIRED ) # For Cuda Managed Memory and c++11

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(.)

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=sm_60)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --ptxas-options=-v --compiler-options -Wall -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75)

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++11)
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-pedantic)
    add_definitions(-Wno-unused-function)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -march=native") # extend release-profile with fast-math
    #add_definitions(-march=native)
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# for YouCompleteMe
set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    fusibile
    cameraGeometryUtils.h
    vector_operations.h
    camera.h
    globalstate.h
    algorithmparameters.h
    cameraparameters.h
    linestate.h
    displayUtils.h
    mathUtils.h
    fileIoUtils.h
    fusibile.cu
    main.cpp
    )

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(fusibile
    ${OpenCV_LIBS}
    )
