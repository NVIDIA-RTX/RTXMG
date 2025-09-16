cmake_minimum_required(VERSION 3.10)

project(rtxmg LANGUAGES C CXX VERSION 1.0.1)

if(WIN32)
    message(STATUS "Found Windows ${CMAKE_SYSTEM_VERSION} - SDK: ${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}")
endif()    

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;ReleaseAsan" CACHE STRING "Configurations" FORCE)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SOURCE_DIR}/extern)

####################################
# Static analysis
####################################
set(RULESET_PATH "${CMAKE_SOURCE_DIR}/rtxmg_static_analysis.ruleset")
add_compile_options("-wd26495")
set_property( GLOBAL PROPERTY VS_USER_PROPS "rtxmg_static_analysis.props")

####################################
# ASAN
####################################
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
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
set(DONUT_STREAMLINE_FETCH_URL "https://github.com/NVIDIA-RTX/Streamline/releases/download/v2.8.0/streamline-sdk-v2.8.0.zip" CACHE STRING "")
set(DONUT_STREAMLINE_FETCH_SHA256 "313669f8cf886f823ea0518a50712efa5e2e8623689ed2a6bd0d9353e475bc47" CACHE STRING "")
set(STREAMLINE_FEATURE_DLSS_SR ON CACHE BOOL "" FORCE)
set(STREAMLINE_FEATURE_DLSS_RR ON CACHE BOOL "" FORCE)
set(STREAMLINE_INSTALL_DIR "bin" CACHE STRING "" FORCE)

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
add_subdirectory(rtxmg)
add_subdirectory(demo)

add_subdirectory(extern)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT rtxmg_demo)
