if (APPLE)
    add_subdirectory(libmudaq)
else()
    add_subdirectory(libmudaq)
    add_subdirectory(kerneldriver)
endif (APPLE)

add_subdirectory(utils)
