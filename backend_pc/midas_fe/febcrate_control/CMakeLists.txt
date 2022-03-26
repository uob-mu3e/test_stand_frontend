cmake_minimum_required(VERSION 3.0)

add_executable(febcratefe febcratefe.cpp)

target_compile_options(febcratefe PRIVATE
	-Wno-missing-field-initializers
	-Wno-unused-parameter
)

target_include_directories(febcratefe PRIVATE ${MIDASSYS}/mscb/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/include)
target_link_libraries(febcratefe mscb mfe midas rt dl pthread util ${Boost_LIBRARIES} libmubanks)

if(DEBUG)
    add_definitions(-DMY_DEBUG)
endif()

install(TARGETS febcratefe DESTINATION bin)
