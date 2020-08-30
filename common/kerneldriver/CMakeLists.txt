cmake_minimum_required(VERSION 3.18)

project(mudaq LANGUAGES C)

file(GLOB MODULE_SOURCES
    ${PROJECT_NAME}.c
    *.h
)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dmabuf/cmake")
find_package(kmodule REQUIRED)
add_kmodule(${PROJECT_NAME} ${MODULE_SOURCES})
