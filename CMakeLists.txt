cmake_minimum_required(VERSION 3.14)
project(SwarmSimulation C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 默认 Release 模式
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# --- 查找基础依赖 ---
find_package(ZLIB REQUIRED)
find_package(OpenMP)

# --- 源文件 ---
add_executable(simulation
    src/main.cpp
    3rdparty/cnpy/cnpy.cpp
)

# --- 头文件包含 ---
target_include_directories(simulation PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/3rdparty/json
    ${CMAKE_SOURCE_DIR}/3rdparty/eigen
    ${CMAKE_SOURCE_DIR}/3rdparty/cnpy
)

# --- 关键定义 ---
target_compile_definitions(simulation PRIVATE _LARGEFILE64_SOURCE)

# =========================================================
# [核心逻辑修复] BLAS / AOCL 互斥查找与链接
# =========================================================

# 1. 优先尝试查找 AMD AOCL (BLIS)
# 注意：NAMES 顺序很重要，优先找多线程版 (blis-mt)
find_library(AOCL_BLAS_LIB NAMES blis-mt blis)

if(AOCL_BLAS_LIB)
    # --- 情况 A: 找到了 AOCL ---
    message(STATUS ">> Found AMD AOCL (BLIS): ${AOCL_BLAS_LIB}")
    target_link_libraries(simulation PRIVATE ${AOCL_BLAS_LIB})
    target_compile_definitions(simulation PRIVATE EIGEN_USE_BLAS)
    message(STATUS ">> Eigen BLAS backend : ENABLED (Using AOCL)")

else()
    # --- 情况 B: 没找到 AOCL，回退到通用 BLAS ---
    message(STATUS ">> AOCL not found, falling back to standard BLAS...")
    find_package(BLAS)

    if(BLAS_FOUND)
        target_link_libraries(simulation PRIVATE ${BLAS_LIBRARIES})
        target_compile_definitions(simulation PRIVATE EIGEN_USE_BLAS)
        message(STATUS ">> Eigen BLAS backend : ENABLED (Using Standard BLAS)")
    endif()
endif()

# =========================================================
# 编译器优化 (Release 模式)
# =========================================================
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # 禁用 Eigen 调试断言
    target_compile_definitions(simulation PRIVATE EIGEN_NO_DEBUG)

    # 激进的编译器优化标志
    target_compile_options(simulation PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-O3 -march=native -funroll-loops -ffast-math>
        $<$<CXX_COMPILER_ID:Clang>:-O3 -march=native -funroll-loops -ffast-math>
    )

    # 开启链接时优化 (LTO/IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if(result)
        set_target_properties(simulation PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS ">> IPO / LTO enabled (Performance Boost!)")
    else()
        message(WARNING "IPO / LTO not supported: ${output}")
    endif()
endif()

# --- 链接其他通用库 ---
target_link_libraries(simulation PRIVATE ${ZLIB_LIBRARIES})
target_link_libraries(simulation PRIVATE stdc++fs) # GCC < 9 兼容

if(OpenMP_CXX_FOUND)
    target_link_libraries(simulation PRIVATE OpenMP::OpenMP_CXX)
endif()

# --- 打印最终状态 ---
message(STATUS "=============================================")
message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler   : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Flags      : -O3 -march=native -ffast-math")
message(STATUS "=============================================")