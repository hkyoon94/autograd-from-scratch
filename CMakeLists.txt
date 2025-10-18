cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
# cmake_policy(SET CMP0148 NEW)

project("autograd-from-scratch")


if (DEFINED ENV{CONDA_PREFIX})
    message(STATUS ">>> Using current conda env")
    set(Python3_ROOT_DIR "$ENV{CONDA_PREFIX}")
else()
    message(STATUS ">>> Not in conda environment, using system packages.")
endif()


# Finding build-sources
message(STATUS ">>> SRC_DIRNAME: ${SRC_DIR}")

file(GLOB_RECURSE CPP_SRC_FILES ${SRC_DIR}/*.cpp)
file(GLOB_RECURSE CUDA_SRC_FILES ${SRC_DIR}/*.cu)
file(GLOB_RECURSE PTX_BINARIES ${SRC_DIR}/*.ptx.o)

list(FILTER CPP_SRC_FILES EXCLUDE REGEX ".*/bindings\\.cpp$")
list(FILTER CPP_SRC_FILES EXCLUDE REGEX ${MAIN_FILENAME})

file(GLOB_RECURSE BINDING_FILES ${SRC_DIR}/bindings*.cpp)

message(STATUS "LIB Impl Build sources:")
foreach(f ${CPP_SRC_FILES} ${CUDA_SRC_FILES} ${PTX_BINARIES})
    file(RELATIVE_PATH rel ${SRC_DIR} ${f})
    message(STATUS "  └─ ${rel}")
endforeach()

message(STATUS "LIB Bind Build sources:")
foreach(f ${BINDING_FILES})
    file(RELATIVE_PATH rel ${SRC_DIR} ${f})
    message(STATUS "  └─ ${rel}")
endforeach()

file(RELATIVE_PATH rel ${SRC_DIR} ${MAIN_FILENAME})
message(STATUS "MAIN source: ${rel}")


if (BUILD_SHARED_LIBS)
    set(LIB ${LIB_FILENAME}_impl)
    add_library(${LIB} STATIC ${CPP_SRC_FILES} ${CUDA_SRC_FILES})
    target_link_libraries(${LIB} PUBLIC ${PTX_BINARIES})
    target_link_options(${LIB} PUBLIC -Wl,-z,noexecstack)
    target_include_directories(${LIB} PUBLIC ${INCLUDE_DIR})
    set_target_properties(${LIB_BIND} PROPERTIES PREFIX "" SUFFIX "")
    
    if (BINDING_FILES)
        set(LIB_BIND ${LIB_FILENAME})
        add_library(${LIB_BIND} SHARED ${BINDING_FILES})
        target_link_libraries(${LIB_BIND} PRIVATE ${LIB})
        target_include_directories(${LIB_BIND} PRIVATE ${INCLUDE_DIR})
        set_target_properties(${LIB_BIND} PROPERTIES PREFIX "" SUFFIX ${EXT_SUFFIX})
    endif()

endif()

if (BUILD_EXE)
    set(EXE ${EXE_FILENAME})
    add_executable(${EXE} ${MAIN_FILENAME})
    target_include_directories(${EXE} PRIVATE ${INCLUDE_DIR})
    if (BUILD_SHARED_LIBS)
        target_link_libraries(${EXE} PRIVATE ${LIB})
    endif()
endif()


# Setting C++ standard for CXX / CUDA compiler
if (BUILD_SHARED_LIBS)
    if (USE_CUDA)
        set_target_properties(${LIB} PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17)
        set_target_properties(${LIB_BIND} PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17)
    else()
        set_target_properties(${LIB} PROPERTIES CXX_STANDARD 17)
        set_target_properties(${LIB_BIND} PROPERTIES CXX_STANDARD 17)
    endif()
endif()
if (BUILD_EXE)
    if (USE_CUDA)
        set_target_properties(${EXE} PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17)
    else()
        set_target_properties(${EXE} PROPERTIES CXX_STANDARD 17)
    endif()
endif()


# Debug / Release compile options
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -pthread")
set(CMAKE_CXX_FLAGS_RELEASE
    "-DNDEBUG \
    -O3 \
    -march=native \
    -mtune=native \
    -fno-fast-math \
    -fstrict-aliasing \
    -fstrict-overflow \
    -fno-merge-all-constants \
    -fno-math-errno \
    -ffp-contract=fast \
    -fomit-frame-pointer \
    -falign-loops=64 \
    -falign-functions=64 \
    -fmerge-all-constants \
    -fno-stack-protector -fno-plt \
    -funroll-loops \
    -fvectorize \
    -fslp-vectorize \
    -fno-trapping-math \
    -fdata-sections -ffunction-sections \
    -flto=thin \
    -fvisibility=hidden \
    -fvisibility-inlines-hidden \
    -fPIC \
    -pipe \
    -pthread \
    -w"
)


# 3rd-party library linking configurations
message(STATUS ">>> USE PYTHON: ${USE_PYTHON}")
message(STATUS ">>> USE CUDA: ${USE_CUDA}")
message(STATUS ">>> USE NUMPY: ${USE_NUMPY}")
message(STATUS ">>> USE TORCH: ${USE_TORCH}")
message(STATUS ">>> USE BLAS: ${USE_BLAS}")


# Pybind11 & Python configuration
if (USE_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    message(STATUS ">>> Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
    message(STATUS ">>> Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
    message(STATUS ">>> Python3_LIBRARIES: ${Python3_LIBRARIES}")
    if (BUILD_SHARED_LIBS)
        target_link_libraries(${LIB} PUBLIC Python3::Python)
        target_link_libraries(${LIB_BIND} PRIVATE Python3::Python)
    endif()
    if (BUILD_EXE)
        target_link_libraries(${EXE} PRIVATE Python3::Python)
    endif()
    find_package(pybind11 CONFIG REQUIRED)
    if (BUILD_SHARED_LIBS)
        target_link_libraries(${LIB} PUBLIC pybind11::module)
        target_link_libraries(${LIB_BIND} PRIVATE pybind11::module)
    endif()
    if (BUILD_EXE)
        target_link_libraries(${EXE} PRIVATE pybind11::module)
    endif()
endif()


# CUDA C++ configuration
if (USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)  # CUDA 라이브러리를 찾기 위한 configuration
    if (BUILD_SHARED_LIBS)
        target_link_libraries(
            ${LIB} PUBLIC CUDA::cublas CUDA::cublasLt CUDA::cudart CUDA::cuda_driver
        )
        set_target_properties(
            ${LIB} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
        )
        target_link_libraries(
            ${LIB_BIND} PRIVATE CUDA::cublas CUDA::cublasLt CUDA::cudart CUDA::cuda_driver
        )
        set_target_properties(
            ${LIB_BIND} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
        )
    endif()
    if (BUILD_EXE)
        target_link_libraries(
            ${EXE} PRIVATE CUDA::cublas CUDA::cublasLt CUDA::cudart CUDA::cuda_driver
        )  # CUDA 라이브러리 링크
        set_target_properties(
            ${EXE} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
        )  # CUDA 컴파일 설정
    endif()
    set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0 -gencode arch=compute_89,code=sm_89 -Xptxas=-v")
    set(CMAKE_CUDA_FLAGS_RELEASE "-DNDEBUG -O3 -gencode arch=compute_89,code=sm_89 -Xptxas=-v")
endif()


# Numpy configuration
if (USE_NUMPY)
    find_package(Python3 COMPONENTS NumPy REQUIRED)
    if (BUILD_SHARED_LIBS)
        target_link_libraries(${LIB} PUBLIC ${NUMPY_LIBRARIES})
        target_link_libraries(${LIB_BIND} PRIVATE ${NUMPY_LIBRARIES})
    endif()
    if (BUILD_EXE)
        target_link_libraries(${EXE} PRIVATE ${NUMPY_LIBRARIES})
    endif()
endif()


# Torch configuration
if (USE_TORCH)
    find_package(Torch REQUIRED)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATH ${TORCH_INSTALL_PREFIX}/lib)
    if (BUILD_SHARED_LIBS)
        target_link_libraries(${LIB} PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
        target_link_libraries(${LIB_BIND} PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
    endif()
    if (BUILD_EXE)
        target_link_libraries(${EXE} PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
    endif()
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
endif()


# BLAS configuration
if (USE_BLAS)
    if (BUILD_SHARED_LIBS)
        target_link_libraries(${LIB} PUBLIC ${BLAS_LIBRARY})
        target_include_directories(${LIB} PUBLIC ${BLAS_INCLUDE_DIR})
        target_link_libraries(${LIB_BIND} PRIVATE ${BLAS_LIBRARY})
        target_include_directories(${LIB} PRIVATE${BLAS_INCLUDE_DIR})
    endif()
    if (BUILD_EXE)
        target_link_libraries(${EXE} PRIVATE ${BLAS_LIBRARY})
        target_include_directories(${EXE} PRIVATE ${BLAS_INCLUDE_DIR})
    endif()
endif()
