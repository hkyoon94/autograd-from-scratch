#!/bin/bash

# --------------------------- >>> CONFIGS >>> --------------------------- #

# Project configurations
MAIN_FILENAME=kernel_test.cpp
SRC_DIRNAME=src
INCLUDE_DIRNAME=src

# Build configurations
BUILD_DIRNAME=build
EXE_FILENAME=main       # output executable filename
LIB_FILENAME=cpp        # output shared library filename (libcpp.so)

# Compiler path configurations
C_COMPILER=/usr/bin/clang
CXX_COMPILER=/usr/bin/clang++

# Python configurations
PYTHON_ENV_PATH=/home/honggyu/miniconda3/envs/work
PYTHON_VER=python3.11

PYTHON_LIBRARY=${PYTHON_ENV_PATH}/lib/lib${PYTHON_VER}.so
PYTHON_INCLUDE_DIR=${PYTHON_ENV_PATH}/include/${PYTHON_VER}

NUMPY_LIBRARY=${PYTHON_ENV_PATH}/lib/${PYTHON_VER}/site-packages/numpy/core
NUMPY_INCLUDE_DIR=${PYTHON_ENV_PATH}/lib/${PYTHON_VER}/site-packages/numpy/core/include

TORCH_CMAKE_PREFIX_PATH=${PYTHON_ENV_PATH}/lib/${PYTHON_VER}/site-packages/torch/share/cmake

# PYTHON_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libpython3.11.so
# PYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
# TORCH_CMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")

# NVCC configurations
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=${CUDA_HOME}/bin:${PATH}
LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

CMAKE_CUDA_ARCHITECTURES=80

# BLAS configurations
# BLAS는 패키지매니저를 통해 제공하지 않기 때문에, 직접 make -> install을 해야함
# make PREFIX=${BLAS_DIR}를 해야 해당 경로에 lib/과 include/가 생성됨
BLAS_DIR=/home/honggyu/openblas  # blas configurations
BLAS_LIBRARY=${BLAS_DIR}/lib/libopenblas.so
BLAS_INCLUDE_DIR=${BLAS_DIR}/include

# --------------------------- <<< CONFIGS <<< --------------------------- #

echo "Python shared lib path: ${PYTHON_LIBRARY}"
echo "Python include dir path: ${PYTHON_INCLUDE_DIR}"
echo "Torch CMake prefix path: ${TORCH_CMAKE_PREFIX_PATH}"
echo "CUDA compiler path: ${CUDA_HOME}"

# Constants
BUILD_TYPE="Debug"
BUILD_SHARED_LIBS=ON
RUN_AFTER_BUILD=false
EXPORT_COMPILE_COMMANDS=OFF
USE_PYTHON=ON
USE_NUMPY=ON
USE_TORCH=ON
USE_CUDA=ON
USE_BLAS=ON

# Parsing flags
for arg in "$@"; do
    case "$arg" in
        -new) rm -rf build;;
        -noshared) BUILD_SHARED_LIBS=OFF; LIB_FILENAME="";;
        -run) RUN_AFTER_BUILD=true;;
        -release) BUILD_TYPE="Release";;
        -nopython) USE_PYTHON=OFF; PYTHON_LIBRARY=""; PYTHON_INCLUDE_DIR="";;
        -nonumpy) USE_NUMPY=OFF; NUMPY_INCLUDE_DIR="";;
        -notorch) USE_TORCH=OFF; TORCH_CMAKE_PREFIX_PATH="";;
        -nocuda) USE_CUDA=OFF; CMAKE_CUDA_ARCHITECTURES="";;
    esac
done

# Making build directory
mkdir -p ${BUILD_DIRNAME}

# Configuration step
cmake -S . -B ${BUILD_DIRNAME} \
    -DCMAKE_C_COMPILER=${C_COMPILER} \
    -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
    -DMAIN_FILENAME=${MAIN_FILENAME} \
    -DSRC_DIRNAME=${SRC_DIRNAME} \
    -DINCLUDE_DIRNAME=${INCLUDE_DIRNAME} \
    -DEXE_FILENAME=${EXE_FILENAME} \
    -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} \
    -DLIB_FILENAME=${LIB_FILENAME} \
    -DUSE_PYTHON=${USE_PYTHON} \
    -DPYTHON_LIBRARY=${PYTHON_LIBRARY} \
    -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} \
    -DUSE_NUMPY=${USE_NUMPY} \
    -DNUMPY_LIBRARY=${NUMPY_LIBRARY} \
    -DNUMPY_INCLUDE_DIR=${NUMPY_INCLUDE_DIR} \
    -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
    -DUSE_TORCH=${USE_TORCH} \
    -DCMAKE_PREFIX_PATH=${TORCH_CMAKE_PREFIX_PATH} \
    -DUSE_CUDA=${USE_CUDA} \
    -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} \
    -DUSE_BLAS=${USE_BLAS} \
    -DBLAS_LIBRARY=${BLAS_LIBRARY} \
    -DBLAS_INCLUDE_DIR=${BLAS_INCLUDE_DIR} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=${EXPORT_COMPILE_COMMANDS}
    # --debug-find

# Building step
cmake --build ${BUILD_DIRNAME} -- -j$(nproc)

touch ${BUILD_DIRNAME}/__init__.py  # For pybind11 shared lib

# mv ./${OUT_FILENAME} ../${OUT_FILENAME}
# mv ./${OUT_FILENAME}.so ../$OUT_FILENAME}.so

if ${RUN_AFTER_BUILD}; then
    echo "Running main executable..."
    ./${BUILD_DIRNAME}/${EXE_FILENAME}
fi
