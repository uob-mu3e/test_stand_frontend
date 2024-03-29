cmake_minimum_required(VERSION 3.8)

project(mudaq-driver LANGUAGES C CXX)

set(MODULE_SOURCES mudaq.c mudaq.h mudaq_fops.h)


list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dmabuf/cmake")
find_package(kmodule REQUIRED)
add_kmodule(${PROJECT_NAME}
    NAME mudaq
    ${MODULE_SOURCES}
)
add_dependencies(${PROJECT_NAME}-insmod ${PROJECT_NAME}-rmmod)

get_directory_property(parent_dir PARENT_DIRECTORY)
if(NOT parent_dir)
    add_custom_target(insmod
        COMMAND sudo chmod a+rw /dev/mudaq0
        COMMAND sudo chmod a+rw /dev/mudaq0_dmabuf
        VERBATIM
        DEPENDS ${PROJECT_NAME}-insmod
    )
endif()
