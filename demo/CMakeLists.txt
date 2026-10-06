# include ("private.cmake" OPTIONAL)

include ("${DONUT_DIR}/compileshaders.cmake")

add_subdirectory(audio)


file(GLOB shaders "shaders/*")
file(GLOB sources "*.cpp" "*.h" "osd_port_sources/tmr/*.h" *.cfg)

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/CMake/FetchImplot.cmake")
    message(FATAL_ERROR "Missing FetchImplot.cmake file")
endif()
include("${PROJECT_SOURCE_DIR}/CMake/FetchImplot.cmake")
set_target_properties(implot PROPERTIES FOLDER "Third-Party Libraries")

set(app rtxmg_demo)
set(folder "Demo")

list(APPEND RTXMG_SHADERS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

donut_compile_shaders_all_platforms(
    TARGET ${app}_shaders
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
    FOLDER ${folder}
    SOURCES ${shaders}
    OUTPUT_BASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/${app}
    SHADERMAKE_OPTIONS ${RTXMG_SHADERMAKE_OPTIONS}
    SHADERMAKE_OPTIONS_SPIRV ${RTXMG_SHADERMAKE_OPTIONS_SPIRV}
    SHADER_MODEL ${RTXMG_SHADERS_SHADERMODEL}
    IGNORE_INCLUDES ${RTXMG_SHADERS_IGNORED_INCLUDES}
    INCLUDES ${RTXMG_SHADERS_INCLUDE_DIR}
)

add_executable(${app} 
    ${sources} 
)
target_link_libraries(${app} PUBLIC envmap rtxmg donut_render donut_app donut_engine implot cgltf)

option(RTXMG_DEV_FEATURES "Enable development features (debugging, etc)" OFF)
if(RTXMG_DEV_FEATURES)
    target_compile_definitions(${app} PUBLIC RTXMG_DEV_FEATURES=1)
endif()
target_compile_definitions(${app} PUBLIC RTXMG_VERSION="${PROJECT_VERSION}")

if(ENABLE_AUDIO_ENGINE)
    if(TARGET audio_engine)
        target_link_libraries(${app} PRIVATE audio_engine)
    else()
        message(WARNING "Audio engine option is enabled, but configuration failed")
    endif()
endif()

add_dependencies(${app} ${app}_shaders)
set_target_properties(${app} PROPERTIES FOLDER ${folder})
add_compile_definitions( "_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING=1")

target_include_directories(${app} PUBLIC 
    ${PROJECT_SOURCE_DIR} 
    ${PROJECT_SOURCE_DIR}/extern
)

add_subdirectory(envmap)

if(DONUT_D3D_AGILITY_SDK_ENABLED)
    target_compile_definitions(${app} PUBLIC DONUT_D3D_AGILITY_SDK_ENABLED=1)
    target_compile_definitions(${app} PUBLIC DONUT_D3D_AGILITY_SDK_VERSION=${DONUT_D3D_AGILITY_SDK_VERSION})
    add_custom_command(TARGET ${app} POST_BUILD
        COMMAND 
            ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${app}>/D3D12/"
        COMMAND 
            ${CMAKE_COMMAND} -E copy_if_different ${DONUT_D3D_AGILITY_SDK_CORE_DLL} "$<TARGET_FILE_DIR:${app}>/D3D12/"
    )
endif()

if(WIN32)
    add_custom_command( TARGET ${app} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:${app}> $<TARGET_FILE_DIR:${app}>
        COMMAND_EXPAND_LISTS)
endif()

#
# Installation & packaging
#
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(RTXMG_ARM64)
        set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/_install_arm64" CACHE PATH "Installation directory" FORCE)
    else()
        set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/_install" CACHE PATH "Installation directory" FORCE)
    endif()
endif()

install(TARGETS ${app} DESTINATION "${RTXMG_BIN_DIR}")

install(DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders" DESTINATION "${RTXMG_BIN_DIR}" )

if(WIN32)
    install(FILES $<TARGET_RUNTIME_DLLS:${app}> DESTINATION "${RTXMG_BIN_DIR}")

    # Donut marks Streamline EXCLUDE_FROM_ALL, which drops its own install rule.
    if(DONUT_WITH_STREAMLINE)
        FetchContent_GetProperties(streamline)
        get_directory_property(_sl_dlls DIRECTORY "${streamline_SOURCE_DIR}" DEFINITION STREAMLINE_DLLS)
        get_directory_property(_sl_jsons DIRECTORY "${streamline_SOURCE_DIR}" DEFINITION STREAMLINE_JSONS)
        install(FILES ${_sl_dlls} ${_sl_jsons} DESTINATION "${RTXMG_BIN_DIR}")
    endif()

    # Scene shortcuts shared by both architectures; a relative -mf resolves under assets/
    function(rtxmg_run_script name RUN_ARGS)
        file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${name}.bat" NEWLINE_STYLE CRLF @ONLY CONTENT [=[
@echo off
set BIN=bin
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" if exist "%~dp0bin_arm64\rtxmg_demo.exe" set BIN=bin_arm64
start "" "%~dp0%BIN%\rtxmg_demo.exe" @RUN_ARGS@
]=])
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${name}.bat" DESTINATION ".")
    endfunction()

    rtxmg_run_script(run_rtxmg "")
    rtxmg_run_script(run_abeautifulgame "-mf cluster_lod/ABeautifulGame/glTF/ABeautifulGame.gltf")
    rtxmg_run_script(run_amy_abeautifulgame "-mf subdivision/amy_abeautifulgame.scene.json")
    rtxmg_run_script(run_amy_kitchenset "-mf subdivision/amy_kitchenset.scene.json")
    rtxmg_run_script(run_barbarian "-mf subdivision/barbarian_pt.scene.json")

    if (RTXMG_ARM64)
        set(_os "win-arm64")
    else()
        set(_os "win64")
    endif()
elseif (UNIX AND NOT APPLE)
    set(_os "linux64")
endif()

if(DONUT_D3D_AGILITY_SDK_ENABLED)
    install(DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/D3D12" DESTINATION "${RTXMG_BIN_DIR}" )
endif()

if(IS_DIRECTORY "${PROJECT_SOURCE_DIR}/assets")
    install(
        DIRECTORY "${PROJECT_SOURCE_DIR}/assets" 
        DESTINATION "." 
        PATTERN ".git*" EXCLUDE
        REGEX "\\.subdcache$" EXCLUDE
        REGEX "nvsngeo" EXCLUDE)
endif()

set(RTXMG_PACKAGE_ARM64_BUILD_DIR "" CACHE PATH "Arm64 build dir whose install is packaged alongside this x64 one")
if(RTXMG_PACKAGE_ARM64_BUILD_DIR AND NOT RTXMG_ARM64)
    set(CPACK_INSTALL_CMAKE_PROJECTS
        "${CMAKE_BINARY_DIR};${CMAKE_PROJECT_NAME};ALL;/"
        "${RTXMG_PACKAGE_ARM64_BUILD_DIR};${CMAKE_PROJECT_NAME};ALL;/")
    set(_os "win-x64-arm64")
endif()

set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}.${PROJECT_VERSION}.${_os}")

include(CPack)
