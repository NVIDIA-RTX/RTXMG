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

donut_compile_shaders(
    TARGET ${app}_shaders
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
    FOLDER ${folder}
    SOURCES ${shaders}
    DXIL ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/${app}/dxil
    # SPIRV_DXC ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/${app}/spirv
    SHADERMAKE_OPTIONS_DXIL ${RTXMG_SHADERMAKE_OPTIONS_DXIL}
    SHADER_MODEL ${RTXMG_SHADERS_SHADERMODEL}
    IGNORE_INCLUDES ${RTXMG_SHADERS_IGNORED_INCLUDES}
    INCLUDES ${RTXMG_SHADERS_INCLUDE_DIR}
)

add_executable(${app} 
    ${sources} 
)
target_link_libraries(${app} PUBLIC envmap rtxmg donut_render donut_app donut_engine implot)

if ( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
    # TBB needed for C++17 parallel algorithms on gcc
    target_link_libraries(${app} PRIVATE TBB::tbb)
endif()

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
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/_install" CACHE PATH "Installation directory" FORCE)
endif()

install(TARGETS ${app} DESTINATION "bin")

install(DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders" DESTINATION "bin" )

if(WIN32)
    install(FILES $<TARGET_RUNTIME_DLLS:${app}> DESTINATION "bin")

    # generate some shortcuts to run various scenes
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/run_rtxmg.bat" "start \"\" bin/rtxmg_demo.exe")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/run_rtxmg.bat" DESTINATION ".")

    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/run_amy_kitchenset.bat" "start \"\" bin/rtxmg_demo.exe -mf amy_kitchenset.scene.json")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/run_amy_kitchenset.bat" DESTINATION ".")

    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/run_barbarian.bat" "start \"\" bin/rtxmg_demo.exe -mf barbarian_pt.scene.json")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/run_barbarian.bat" DESTINATION ".")

    set(_os "win64")
elseif (UNIX AND NOT APPLE)
    set(_os "linux64")
endif()

if(DONUT_D3D_AGILITY_SDK_ENABLED)
    install(DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/D3D12" DESTINATION "bin" )
endif()

if(IS_DIRECTORY "${PROJECT_SOURCE_DIR}/assets")
    install(
        DIRECTORY "${PROJECT_SOURCE_DIR}/assets" 
        DESTINATION "." 
        PATTERN "*.bin" EXCLUDE)
endif()

set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}.${PROJECT_VERSION}.${_os}")

include(CPack)
