
if(NOT MIDAS_FOUND)
    add_subdirectory(midas)
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/midas)
    find_package(Midas REQUIRED)
    add_subdirectory(midas/mscb)
endif()
if(ROOT_FOUND)
    add_subdirectory(analyzer)
else()
    message("No ROOT found, analyzer will not be build")
endif()


