cmake_minimum_required(VERSION 3.18)

project(kerneldriver LANGUAGES C)

set(MODULE_NAME mudaq)

execute_process(OUTPUT_VARIABLE KERNEL_RELEASE
    COMMAND uname --kernel-release
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(KERNEL_RELEASE STREQUAL "")
    message(FATAL_ERROR "E [] `uname --kernel-release`")
endif()

set(KDIR "/lib/modules/${KERNEL_RELEASE}/build")
message("I [] KDIR = '${KDIR}'")

file(GLOB MODULE_SOURCES
    ${MODULE_NAME}.c
    *.h
)

add_custom_command(OUTPUT ${MODULE_NAME}.ko
    COMMAND $(MAKE) -C ${KDIR} clean modules
        M=${CMAKE_CURRENT_BINARY_DIR} src=${CMAKE_CURRENT_SOURCE_DIR}
        -E "MODULE_NAME := ${MODULE_NAME}"
    VERBATIM
    DEPENDS Kbuild ${MODULE_SOURCES}
)

add_custom_target(${MODULE_NAME}_module
    ALL
    DEPENDS ${MODULE_NAME}.ko
)



add_custom_target(${MODULE_NAME}_insmod
    COMMAND sudo insmod ${MODULE_NAME}.ko
    COMMAND sudo chmod a+rw /dev/${MODULE_NAME}
    DEPENDS ${MODULE_NAME}.ko
)

add_custom_target(${MODULE_NAME}_rmmod
    COMMAND sudo rmmod ${MODULE_NAME}
)
