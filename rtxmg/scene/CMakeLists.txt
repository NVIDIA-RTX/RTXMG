set(lib scene)

file(GLOB sources "*.cpp" "../include/rtxmg/${lib}/*.h")

add_library(${lib} OBJECT ${sources})
target_include_directories(${lib} PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)
target_link_libraries(${lib} donut_engine osd_lite_static cluster_lod)

# libstdc++ runs std::execution policies serially unless TBB is available.
if(NOT WIN32)
    find_package(TBB REQUIRED)
    target_link_libraries(${lib} TBB::tbb)
endif()
set_target_properties(${lib} PROPERTIES FOLDER RTXMG)
