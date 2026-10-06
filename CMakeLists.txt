cmake_minimum_required(VERSION 4.1)

project(rtxmg LANGUAGES C CXX VERSION 2.1.0)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;ReleaseAsan" CACHE STRING "Configurations" FORCE)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SOURCE_DIR}/extern)

# CMAKE_SYSTEM_PROCESSOR is the host in a Visual Studio cross build; the compiler knows the target.
if(CMAKE_CXX_COMPILER_ARCHITECTURE_ID MATCHES "^(ARM64|arm64|aarch64)$")
    set(RTXMG_ARM64 ON)
endif()

####################################
# Static analysis
####################################
set(RULESET_PATH "${CMAKE_SOURCE_DIR}/rtxmg_static_analysis.ruleset")
add_compile_options("-wd26495")
set_property( GLOBAL PROPERTY VS_USER_PROPS "rtxmg_static_analysis.props")

####################################
# ASAN
####################################
# Also the install folder, so one package can carry both architectures next to a shared assets/.
if(RTXMG_ARM64)
    set(RTXMG_BIN_DIR "bin_arm64")
else()
    set(RTXMG_BIN_DIR "bin")
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/${RTXMG_BIN_DIR}")
foreach(CONFIG_TYPE DEBUG RELEASE MINSIZEREL RELWITHDEBINFO RELEASEASAN)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_TYPE} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endforeach()
set(CMAKE_CXX_FLAGS_RELEASEASAN "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fsanitize=address")
set(CMAKE_EXE_LINKER_FLAGS_RELEASEASAN "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO}")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASEASAN "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO}")

####################################
# AgilitySDK
####################################
if(WIN32)
    # this SDK requires features from the D3D12 Agility SDK on Win 10
    if(CMAKE_SYSTEM_VERSION LESS 11.0)
        set(DONUT_D3D_AGILITY_SDK_URL "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/1.614.1")
        # set fetch destination to cmake default build/_deps folder
        set(DONUT_D3D_AGILITY_SDK_FETCH_DIR "" CACHE STRING "")
        include("${PROJECT_SOURCE_DIR}/extern/donut/cmake/FetchAgilitySDK.cmake")
    else()
        message(STATUS "Windows 11 or newer detected: disabling Agility SDK")
    endif()
endif()

####################################
# DXC
# WindowsSDK version is buggy and the compiler crashes
####################################
set(SHADERMAKE_FIND_DXC ON CACHE BOOL "" FORCE)

####################################
# Streamline
####################################
set(DONUT_WITH_STREAMLINE ON CACHE BOOL "" FORCE)
set(STREAMLINE_FEATURE_DLSS_SR ON CACHE BOOL "" FORCE)
set(STREAMLINE_FEATURE_DLSS_RR ON CACHE BOOL "" FORCE)
set(STREAMLINE_INSTALL_DIR "${RTXMG_BIN_DIR}" CACHE STRING "" FORCE)

####################################
# NVAPI
####################################
include("${PROJECT_SOURCE_DIR}/CMake/FetchNVAPI.cmake")

####################################
# DONUT
####################################
option(DONUT_WITH_DX11 OFF)
set(DONUT_WITH_VULKAN ON CACHE BOOL "Enable the Vulkan version of Donut")
set(NVRHI_WITH_NVAPI ON CACHE BOOL "" FORCE)
set(DONUT_WITH_AFTERMATH ON CACHE BOOL "" FORCE)
set(NVRHI_INSTALL OFF CACHE BOOL "" FORCE)

# Shaders
set(DONUT_SHADERS_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/framework")
set(DONUT_DIR "${PROJECT_SOURCE_DIR}/extern/donut")

list(APPEND RTXMG_SHADERS_INCLUDE_DIR
"${PROJECT_SOURCE_DIR}"
"${PROJECT_SOURCE_DIR}/extern"
"${PROJECT_SOURCE_DIR}/extern/donut/include"
"${PROJECT_SOURCE_DIR}/rtxmg/include"
"${nvapi_SOURCE_DIR}"
"${PROJECT_SOURCE_DIR}/extern/donut/nvrhi/include")

set(RTXMG_SHADERMAKE_OPTIONS "--embedPDB --hlsl2021")

#-vd disable validator due to a bug in 
# https://github.com/microsoft/DirectXShaderCompiler/issues/7181
# should be fixed in version 1.8.2505
set(RTXMG_SHADERMAKE_OPTIONS_SPIRV "--vulkanMemoryLayout scalar --compilerOptions -Vd") 
set(RTXMG_SHADERS_SHADERMODEL "6_6")

set(RTXMG_SHADERS_IGNORED_INCLUDES "")
list(APPEND RTXMG_SHADERS_IGNORED_INCLUDES
    "donut/core/math/math.h"
    "nvrhi/nvrhi.h"
    "array"
    "assert.h"
    "cstdint"
    "cstdio"
    "stdio.h"
    "cmath"
    "ostream"
    "rtxmg/utils/buffer.h"
)

add_subdirectory(extern/donut)
add_subdirectory(extern/osd_lite EXCLUDE_FROM_ALL)

####################################
# meshoptimizer (for cluster_lod baker)
####################################
include(FetchContent)
set(MESHOPT_BUILD_DEMO OFF CACHE BOOL "" FORCE)
set(MESHOPT_BUILD_GLTFPACK OFF CACHE BOOL "" FORCE)
set(MESHOPT_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    meshoptimizer
    GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
    GIT_TAG        v1.1
)
FetchContent_MakeAvailable(meshoptimizer)
set_target_properties(meshoptimizer PROPERTIES FOLDER "extern")

####################################
# mimalloc — scalable allocator used ONLY by meshoptimizer (via
# meshopt_setAllocator) to avoid global-heap-lock contention during the
# parallel cluster-LOD bake. Not a global new/delete override.
####################################
set(MI_BUILD_SHARED   OFF CACHE BOOL "" FORCE)
set(MI_BUILD_OBJECT   OFF CACHE BOOL "" FORCE)
set(MI_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(MI_OVERRIDE       OFF CACHE BOOL "" FORCE)  # do NOT replace global malloc/new
FetchContent_Declare(
    mimalloc
    GIT_REPOSITORY https://github.com/microsoft/mimalloc.git
    GIT_TAG        v2.1.7
)
FetchContent_MakeAvailable(mimalloc)
set_target_properties(mimalloc-static PROPERTIES FOLDER "extern")
# mimalloc installs its headers, .lib and cmake/pkgconfig files, which would
# otherwise land in the package as include/ and lib/.
set_property(DIRECTORY "${mimalloc_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL YES)

# The C++ and the HLSL halves of the ring have to agree, and the shader compile
# takes its defines from the ShaderMake option string, not from CMake targets.
option(RTXMG_SHADER_DEBUG "Compile in the shader-debug print ring (shader_debug.h)" ON)
if(NOT RTXMG_SHADER_DEBUG)
    add_compile_definitions(ENABLE_SHADER_DEBUG=0)
    string(APPEND RTXMG_SHADERMAKE_OPTIONS " -D ENABLE_SHADER_DEBUG=0")
endif()

add_subdirectory(rtxmg)
add_subdirectory(demo)

add_subdirectory(extern)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT rtxmg_demo)
