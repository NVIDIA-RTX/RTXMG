#
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

if( TARGET implot )
    return()
endif()


if (NOT TARGET imgui)
    message(FATAL_ERROR "Implot requires imgui")
endif()

include(FetchContent)
FetchContent_Declare(
    implot
    GIT_REPOSITORY https://github.com/epezent/implot.git
    GIT_TAG v0.17
    )
FetchContent_MakeAvailable(implot)

# Override Imgui build - we want a lean static library

set(implot_srcs
    ${CMAKE_BINARY_DIR}/_deps/implot-src/implot.cpp
    ${CMAKE_BINARY_DIR}/_deps/implot-src/implot.h
    ${CMAKE_BINARY_DIR}/_deps/implot-src/implot_internal.h
    ${CMAKE_BINARY_DIR}/_deps/implot-src/implot_items.cpp
)

add_library(implot STATIC ${implot_srcs})
set_target_properties(implot PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(implot PUBLIC IMGUI_DEFINE_MATH_OPERATORS)
target_include_directories(implot PUBLIC "${CMAKE_BINARY_DIR}/_deps/implot-src/")
target_link_libraries(implot imgui)
