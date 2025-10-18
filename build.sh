#!/bin/bash

# Build args (Base build mode is 'DEBUG')
#   -new: Build from scratch
#   -noshared: Disables off shared-lib linking
#   -run: Runs executable after building
#   -release: Release mode build (nodebug)

# --------------------------- >>> PROJECT CONFIGS >>> ---------------------------

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Project & CXX build configurations
PROJECT_ROOT=$(pwd)
PKG_NAME=autograd

SRC_DIR=${PROJECT_ROOT}/${PKG_NAME}/csrc
INCLUDE_DIR=${SRC_DIR}
MAIN_FILENAME=${SRC_DIR}/main.cpp
PTXGET_PATH=${SRC_DIR}/triton/ptxget.py

LIB_FILENAME=_C  # output shared library filename (_C.so)
EXE_FILENAME=main  # output executable filename
BUILD_DIRNAME=.build

USE_PYTHON=ON  # can manually turn on/off external lib include & linking
USE_CUDA=ON
USE_NUMPY=OFF
USE_TORCH=OFF
USE_BLAS=OFF

# --------------------------- >>> LIBRARY CONFIGS >>> ---------------------------

# Virtual env configuration
ENV_PATH=${HOME}/miniconda3/envs/ai

# Compiler path configurations
C_COMPILER=/usr/bin/clang
CXX_COMPILER=/usr/bin/clang++

# Python configurations
PYTHON_VER=3.12
SITE_PKGS_PATH=${ENV_PATH}/lib/python${PYTHON_VER}/site-packages
export PYTHONPATH=${PROJECT_ROOT}

PYTHON_LIBRARY=${ENV_PATH}/lib/libpython${PYTHON_VER}.so
PYBIND11_DIR=${SITE_PKGS_PATH}/pybind11/share/cmake/pybind11

# NVCC configurations
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
CUDA_COMPILER=$(which nvcc)
CMAKE_CUDA_ARCHITECTURES=89

# Numpy
NUMPY_LIBRARY=${SITE_PKGS_PATH}/numpy/core
NUMPY_INCLUDE_DIR=${SITE_PKGS_PATH}/numpy/core/include

# Torch
TORCH_CMAKE_PREFIX_PATH=${SITE_PKGS_PATH}/torch/share/cmake

# BLAS configurations
# BLAS는 패키지매니저를 통해 제공하지 않기 때문에, 직접 make -> install을 해야함
# make PREFIX=${BLAS_DIR}를 해야 해당 경로에 lib/과 include/가 생성됨
BLAS_DIR=${BLAS_DIR}/openblas  # blas configurations
BLAS_LIBRARY=${BLAS_DIR}/lib/libopenblas.so
BLAS_INCLUDE_DIR=${BLAS_DIR}/include


# ---------------------------- Build & Installing ----------------------------

# Parsing flags

# Constants
DELETE_BUILD_CACHE=false
BUILD_TYPE="Debug"
BUILD_SHARED_LIBS=ON
BUILD_EXE=ON
EXPORT_COMPILE_COMMANDS=OFF
BUILD_WHEEL=false
INSTALL_AND_TEST=false
BUILD_TEST_FILES=false

for arg in "$@"; do
    case "$arg" in
        -new) DELETE_BUILD_CACHE=true;;
        -noshared) BUILD_SHARED_LIBS=OFF;;
        -noexe) BUILD_EXE=OFF;;
        -release) BUILD_TYPE="Release";;
        -wheel) BUILD_WHEEL=true;;
        -install) INSTALL_AND_TEST=true;;
        -t) BUILD_TEST_FILES=true;;
        *)
            if ${BUILD_TEST_FILES}; then
                TEST_FILE=$arg
            fi
            ;;
    esac
done


if ${BUILD_WHEEL}; then
    rm -rf .wheel
    mkdir -p .wheel
    find . -type d -name "__pycache__" -exec rm -r {} +
    python -m build -o .wheel
    rm -rf *egg-info
    pip install .wheel/*.whl --target .wheel/
    tree .wheel/${PKG_NAME} -I "__pycache__" # can include __pycache__ after unzipping

elif ${INSTALL_AND_TEST}; then
    {
        set -e
        pip install .wheel/*.whl
        cd ${SITE_PKGS_PATH}/${PKG_NAME}
        pytest
    }
    pip uninstall ${PKG_NAME}
    # Need to exclude 'tests/*' for strict release

else
    echo "------------------------ Building CXX sources ------------------------"
    
    set -e
    # Compiling & extracting ptx kernels via Triton
    python ${PTXGET_PATH}

    echo "Python .so path: ${PYTHON_LIBRARY}"
    echo "Python headers path: ${PYTHON_INCLUDE_DIR}"
    echo "Pybind11 CMake prefix path: ${PYBIND11_DIR}"
    echo "Torch CMake prefix path: ${TORCH_CMAKE_PREFIX_PATH}"
    echo "CUDA compiler path: ${CUDA_HOME}"

    NPROCS=1
    # NPROCS=$(($(nproc) - 2))
    echo "***** # Procs for parallel-build: ${NPROCS}"

    # --------------------------- <<< CMAKE CONFIGS <<< ---------------------------

    EXT_SUFFIX=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

    if ${BUILD_TEST_FILES}; then
        SRC_DIRNAME=.test
        BUILD_SHARED_LIBS=OFF
        MAIN_FILENAME=${PROJECT_ROOT}/${TEST_FILE}
        EXE_FILENAME=test
        BUILD_DIRNAME=.build-test
    fi

    CMAKE_ARGS="
        -DPKG_NAME=${PKG_NAME} \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DEXT_SUFFIX=${EXT_SUFFIX} \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=${EXPORT_COMPILE_COMMANDS} \

        -DSRC_DIR=${SRC_DIR} \
        -DINCLUDE_DIR=${INCLUDE_DIR} \
        
        -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} \
        -DLIB_FILENAME=${LIB_FILENAME} \

        -DBUILD_EXE=${BUILD_EXE} \
        -DMAIN_FILENAME=${MAIN_FILENAME} \
        -DEXE_FILENAME=${EXE_FILENAME} \

        -DCMAKE_C_COMPILER=${C_COMPILER} \
        -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \

        -DUSE_PYTHON=${USE_PYTHON} \
        -Dpybind11_DIR=${PYBIND11_DIR} \

        -DUSE_CUDA=${USE_CUDA} \
        -DCMAKE_CUDA_COMPILER=${CUDA_COMPILER} \
        -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} \

        -DUSE_NUMPY=${USE_NUMPY} \
        -DNUMPY_LIBRARY=${NUMPY_LIBRARY} \
        -DNUMPY_INCLUDE_DIR=${NUMPY_INCLUDE_DIR} \
        -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \

        -DUSE_TORCH=${USE_TORCH} \
        -DCMAKE_PREFIX_PATH=${TORCH_CMAKE_PREFIX_PATH} \

        -DUSE_BLAS=${USE_BLAS} \
        -DBLAS_LIBRARY=${BLAS_LIBRARY} \
        -DBLAS_INCLUDE_DIR=${BLAS_INCLUDE_DIR}
    "

    echo "----------------------------- CMake args -----------------------------"
    args=($CMAKE_ARGS)
    for arg in "${args[@]}"; do
        echo "$arg"
    done
    echo "----------------------------------------------------------------------"

    if ${DELETE_BUILD_CACHE}; then
        rm -rf ${BUILD_DIRNAME}
    fi
    cmake -S . -B ${BUILD_DIRNAME} ${CMAKE_ARGS}
    cmake --build ${BUILD_DIRNAME} -- -j${NPROCS}

    ln -sfn $(realpath ${BUILD_DIRNAME}/${LIB_FILENAME}.*.so) ./${PKG_NAME}/lib/
    echo "Created symlink ${BUILD_DIRNAME}/*.so -> ./${PKG_NAME}/lib/*.so"

    if [ "${BUILD_EXE}" = "ON" ]; then
        echo "Running main executable..."
        ./${BUILD_DIRNAME}/${EXE_FILENAME}
    fi
    set +e
fi
