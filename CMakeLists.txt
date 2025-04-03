cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project("my_project")

# ${SRC_FILENAME}/ 내에 있는 모든 .cpp & .cu 파일 경로 파싱
message("***** CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message("***** SRC_DIRNAME: ${SRC_DIRNAME}")
file(GLOB_RECURSE CPP_SRC_FILES "${CMAKE_SOURCE_DIR}/${SRC_DIRNAME}/*.cpp")
if (USE_CUDA)
    file(GLOB_RECURSE CUDA_SRC_FILES "${CMAKE_SOURCE_DIR}/${SRC_DIRNAME}/*.cu")
endif()


# 실행파일 및 공유 라이브러리리 빌드 설정
if (CPP_SRC_FILES AND CUDA_SRC_FILES)
    add_executable("${EXE_FILENAME}" "${MAIN_FILENAME}" "${CPP_SRC_FILES}" "${CUDA_SRC_FILES}")
    if (BUILD_SHARED_LIBS)
        add_library("${LIB_FILENAME}" SHARED "${CPP_SRC_FILES}" "${CUDA_SRC_FILES}")
    endif()
elseif (CPP_SRC_FILES)
    add_executable("${EXE_FILENAME}" "${MAIN_FILENAME}" "${CPP_SRC_FILES}")
    if (BUILD_SHARED_LIBS)
        add_library("${LIB_FILENAME}" SHARED "${CPP_SRC_FILES}")
    endif()
elseif (CUDA_SRC_FILES)
    add_executable("${EXE_FILENAME}" "${MAIN_FILENAME}" "${CUDA_SRC_FILES}")
    if (BUILD_SHARED_LIBS)
        add_library("${LIB_FILENAME}" SHARED "${CUDA_SRC_FILES}")
    endif()
else()
    add_executable("${EXE_FILENAME}" "${MAIN_FILENAME}")
    set(BUILD_SHARED_LIBS OFF)
endif()


# 헤더파일 경로 설정 (지정하지 않으면 헤더파일이 소스파일과 같은 디렉토리에 위치하는 경우)
option(INCLUDE_DIRECTORIES "include" "include")
message("***** INCLUDE_DIRNAME: ${INCLUDE_DIRNAME}")
if (INCLUDE_DIRNAME)
    include_directories("${CMAKE_SOURCE_DIR}/${INCLUDE_DIRNAME}")
endif()


# C++ 표준을 C++17로 설정
set_property(TARGET "${EXE_FILENAME}" PROPERTY CXX_STANDARD 17)
if (BUILD_SHARED_LIBS)
    set_property(TARGET "${LIB_FILENAME}" PROPERTY CXX_STANDARD 17)
endif()


# Debug와 Release 모드 컴파일 설정
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -ffp-contract=off -flto -funroll-loops -pthread -ffp-model=precise")


# 외부 라이브러리 사용을 위한 configuration
option(USE_PYTHON "Use Python" ON)
option(USE_NUMPY "Use Numpy" ON)
option(USE_TORCH "Use Torch" ON)
option(USE_CUDA "Use CUDA" ON)
option(USE_BLAS "Use OpenBLAS" ON)
message("***** USE PYTHON: ${USE_PYTHON}")
message("***** USE NUMPY: ${USE_NUMPY}")
message("***** USE TORCH: ${USE_TORCH}")
message("***** USE CUDA: ${USE_CUDA}")
message("***** USE BLAS: ${USE_CBLAS}")


# Python 사용을 위한 configuration
if (USE_PYTHON)
    find_package(PythonLibs REQUIRED)
    include_directories("${PYTHON_INCLUDE_DIRS}")
    target_link_libraries("${EXE_FILENAME}" "${PYTHON_LIBRARIES}")
    if (BUILD_SHARED_LIBS)
        target_link_libraries("${LIB_FILENAME}" "${PYTHON_LIBRARIES}")
    endif()
endif()


# Numpy 사용을 위한 configuration
if (USE_NUMPY)
    find_package(Python3 COMPONENTS NumPy REQUIRED)
    include_directories("${Python3_NumPy_INCLUDE_DIRS}")
    target_link_libraries("${EXE_FILENAME}" "${NUMPY_LIBRARIES}")
    if (BUILD_SHARED_LIBS)
        target_link_libraries("${LIB_FILENAME}" "${NUMPY_LIBRARIES}")
    endif()
endif()


# Torch 컴파일을 위한 configuration
if (USE_TORCH)
    find_package(Torch REQUIRED)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
    target_link_libraries("${EXE_FILENAME}" ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
    if (BUILD_SHARED_LIBS)
        target_link_libraries("${LIB_FILENAME}" ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()


# CUDA C++ 컴파일을 위한 configuration
if (USE_CUDA)
    find_package(CUDA REQUIRED)  # CUDA 라이브러리를 찾기 위한 configuration
    include_directories("${CUDA_INCLUDE_DIRS}")  # CUDA 헤더 파일 경로 설정
    target_link_libraries("${EXE_FILENAME}" "${CUDA_LIBRARIES}" cublas cublasLt cudart)  # CUDA 라이브러리 링크
    set_target_properties("${EXE_FILENAME}" PROPERTIES CUDA_SEPARABLE_COMPILATION ON)  # CUDA 컴파일 설정
    if (BUILD_SHARED_LIBS)
        target_link_libraries("${LIB_FILENAME}" "${CUDA_LIBRARIES}" cublas cublasLt cudart)
        set_target_properties("${LIB_FILENAME}" PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif()
    set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")
    set(CMAKE_CUDA_FLAGS_RELEASE"-O3 -gencode arch=compute_80,code=sm_75 -Xptxas")
endif()


# BLAS 사용을 위한 configuration
if (USE_BLAS)
    include_directories("${BLAS_INCLUDE_DIR}")
    target_link_libraries("${EXE_FILENAME}" "${BLAS_LIBRARY}")
    if (BUILD_SHARED_LIBS)
        target_link_libraries("${LIB_FILENAME}" "${BLAS_LIBRARY}")
    endif()
endif()
