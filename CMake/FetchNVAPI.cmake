#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
include(FetchContent)

set(NVAPI_FETCH_URL "https://github.com/NVIDIA/nvapi.git" CACHE STRING "Url to nvapi git repo to fetch")
set(NVAPI_FETCH_TAG "87dca625e83fd89a983e19b904e5f3a580da90d2" CACHE STRING "Tag of nvapi git repo")
set(NVAPI_FETCH_DIR "" CACHE STRING "Directory to fetch streamline to, empty uses build directory default")

include(FetchContent)
FetchContent_Declare(
    nvapi
    GIT_REPOSITORY ${NVAPI_FETCH_URL}
    GIT_TAG ${NVAPI_FETCH_TAG}
    SOURCE_DIR ${NVAPI_FETCH_DIR}
)
FetchContent_MakeAvailable(nvapi)

message(STATUS "Updating nvapi from ${NVAPI_FETCH_URL}, tag ${NVAPI_FETCH_TAG}, into folder ${nvapi_SOURCE_DIR}")
set(NVAPI_SEARCH_PATHS "${nvapi_SOURCE_DIR}")