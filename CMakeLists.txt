cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
# cmake_policy(SET CMP0148 NEW)

project("autograd-from-scratch")


# Finding build-sources
message("***** CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message("***** SRC_DIRNAME: ${SRC_DIRNAME}")

file(GLOB_RECURSE CPP_SRC_FILES "${CMAKE_SOURCE_DIR}/${SRC_DIRNAME}/*.cpp")
if (USE_CUDA)
    file(GLOB_RECURSE CUDA_SRC_FILES "${CMAKE_SOURCE_DIR}/${SRC_DIRNAME}/*.cu")
    file(GLOB_RECURSE PTX_BINARIES "${CMAKE_SOURCE_DIR}/${SRC_DIRNAME}/*.ptx.o")
endif()

message(STATUS "Build sources:")
foreach(f ${CPP_SRC_FILES} ${CUDA_SRC_FILES} ${PTX_BINARIES})
    file(RELATIVE_PATH rel ${CMAKE_SOURCE_DIR} ${f})
    message(STATUS "  └─ ${rel}")
endforeach()


# Defining executable & .so lib scope
if (CPP_SRC_FILES AND CUDA_SRC_FILES)
    if (BUILD_EXE)
        add_executable("${EXE_FILENAME}" "${MAIN_FILENAME}" "${CPP_SRC_FILES}" "${CUDA_SRC_FILES}")
    endif()
    if (BUILD_SHARED_LIBS)
        add_library("${LIB_FILENAME}" SHARED "${CPP_SRC_FILES}" "${CUDA_SRC_FILES}")
    endif()
elseif (CPP_SRC_FILES)
    if (BUILD_EXE)
        add_executable("${EXE_FILENAME}" "${MAIN_FILENAME}" "${CPP_SRC_FILES}")
    endif()
    if (BUILD_SHARED_LIBS)
        add_library("${LIB_FILENAME}" SHARED "${CPP_SRC_FILES}")
    endif()
elseif (CUDA_SRC_FILES)
    if (BUILD_EXE)
        add_executable("${EXE_FILENAME}" "${MAIN_FILENAME}" "${CUDA_SRC_FILES}")
    endif()
    if (BUILD_SHARED_LIBS)
        add_library("${LIB_FILENAME}" SHARED "${CUDA_SRC_FILES}")
    endif()
else()
    if (BUILD_EXE)
        add_executable("${EXE_FILENAME}" "${MAIN_FILENAME}")
    endif()
    set(BUILD_SHARED_LIBS OFF)
endif()


# Linking pre-built .ptx.o binaries
if (USE_CUDA)
    target_link_libraries("${EXE_FILENAME}" PRIVATE "${PTX_BINARIES}")
    target_link_options("${EXE_FILENAME}" PRIVATE -Wl,-z,noexecstack)
    if (BUILD_SHARED_LIBS)
        target_link_libraries("${LIB_FILENAME}" PRIVATE "${PTX_BINARIES}")
        target_link_options("${LIB_FILENAME}" PRIVATE -Wl,-z,noexecstack)
    endif ()
endif ()


# Setting PIP-standard wheel suffix to .so lib
set_target_properties(${LIB_FILENAME} PROPERTIES PREFIX "" SUFFIX "${EXT_SUFFIX}")

message("***** INCLUDE_DIRNAME: ${INCLUDE_DIRNAME}")
if (BUILD_EXE)
    target_include_directories(
        "${EXE_FILENAME}" PRIVATE "${CMAKE_SOURCE_DIR}/${INCLUDE_DIRNAME}"
    )
endif()
if (BUILD_SHARED_LIBS)
    target_include_directories(
        "${LIB_FILENAME}" PRIVATE "${CMAKE_SOURCE_DIR}/${INCLUDE_DIRNAME}"
    )
endif()


# Setting C++ standard for CXX / CUDA compiler
if (BUILD_EXE)
    if (USE_CUDA)
        set_target_properties(
            ${EXE_FILENAME} PROPERTIES
            CXX_STANDARD 17
            CUDA_STANDARD 17
        )
    else()
        set_target_properties(
            ${EXE_FILENAME} PROPERTIES
            CXX_STANDARD 17
        )
    endif()
endif()
if (BUILD_SHARED_LIBS)
    if (USE_CUDA)
        set_target_properties(
            ${LIB_FILENAME} PROPERTIES
            CXX_STANDARD 17
            CUDA_STANDARD 17
        )
    else()
        set_target_properties(
            ${LIB_FILENAME} PROPERTIES
            CXX_STANDARD 17
        )
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
    -pipe \
    -pthread \
    -w"
)

# add_compile_options(
#   -ferror-limit=1
#   -ftemplate-backtrace-limit=0
#   --gcc-toolchain=/usr
# )


# 3rd-party library linking configurations
message("***** USE PYTHON: ${USE_PYTHON}")
message("***** USE CUDA: ${USE_CUDA}")
message("***** USE NUMPY: ${USE_NUMPY}")
message("***** USE TORCH: ${USE_TORCH}")
message("***** USE BLAS: ${USE_BLAS}")


# Pybind11 & Python configuration
if (USE_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    if (BUILD_EXE)
        target_link_libraries(
            "${EXE_FILENAME}" PRIVATE Python3::Python
        )
    endif()
    if (BUILD_SHARED_LIBS)
        target_link_libraries(
            "${LIB_FILENAME}" PRIVATE Python3::Python
        )
    endif()
    find_package(pybind11 CONFIG REQUIRED)
    if (BUILD_EXE)
        target_link_libraries(
            "${EXE_FILENAME}" PRIVATE pybind11::module
        )
    endif()
    if (BUILD_SHARED_LIBS)
        target_link_libraries(
            "${LIB_FILENAME}" PRIVATE pybind11::module
        )
    endif()
endif()


# CUDA C++ configuration
if (USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)  # CUDA 라이브러리를 찾기 위한 configuration
    if (BUILD_EXE)
        target_link_libraries(
            "${EXE_FILENAME}" PRIVATE CUDA::cublas CUDA::cublasLt CUDA::cudart CUDA::cuda_driver
        )  # CUDA 라이브러리 링크
        set_target_properties(
            "${EXE_FILENAME}" PROPERTIES CUDA_SEPARABLE_COMPILATION ON
        )  # CUDA 컴파일 설정
    endif()
    if (BUILD_SHARED_LIBS)
        target_link_libraries(
            "${LIB_FILENAME}" PRIVATE CUDA::cublas CUDA::cublasLt CUDA::cudart CUDA::cuda_driver
        )
        set_target_properties(
            "${LIB_FILENAME}" PROPERTIES CUDA_SEPARABLE_COMPILATION ON
        )
    endif()
    set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0 -gencode arch=compute_89,code=sm_89 -Xptxas=-v")
    set(CMAKE_CUDA_FLAGS_RELEASE "-DNDEBUG -O3 -gencode arch=compute_89,code=sm_89 -Xptxas=-v")
endif()


# Numpy configuration
if (USE_NUMPY)
    find_package(Python3 COMPONENTS NumPy REQUIRED)
    if (BUILD_EXE)
        target_include_directories(
            "${EXE_FILENAME}" PRIVATE "${Python3_NumPy_INCLUDE_DIRS}"
        )
        target_link_libraries(
            "${EXE_FILENAME}" PRIVATE "${NUMPY_LIBRARIES}"
        )
    endif()
    if (BUILD_SHARED_LIBS)
        target_include_directories(
            "${LIB_FILENAME}" PRIVATE "${Python3_NumPy_INCLUDE_DIRS}"
        )
        target_link_libraries(
            "${LIB_FILENAME}" PRIVATE "${NUMPY_LIBRARIES}"
        )
    endif()
endif()


# Torch configuration
if (USE_TORCH)
    find_package(Torch REQUIRED)
    find_library(
        TORCH_PYTHON_LIBRARY torch_python PATH ${TORCH_INSTALL_PREFIX}/lib
    )
    if (BUILD_EXE)
        target_link_libraries(
            ${EXE_FILENAME} PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY}
        )
    endif()
    if (BUILD_SHARED_LIBS)
        target_link_libraries(
            ${LIB_FILENAME} PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY}
        )
    endif()
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
endif()


# BLAS configuration
if (USE_BLAS)
    include_directories("${BLAS_INCLUDE_DIR}")
    if (BUILD_EXE)
        target_link_libraries("${EXE_FILENAME}" "${BLAS_LIBRARY}")
    endif()
    if (BUILD_SHARED_LIBS)
        target_link_libraries("${LIB_FILENAME}" "${BLAS_LIBRARY}")
    endif()
endif()
