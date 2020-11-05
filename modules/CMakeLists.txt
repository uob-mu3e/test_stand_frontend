
if(NOT MIDAS_FOUND)
    add_subdirectory(midas)
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/midas)
    find_package(Midas REQUIRED)
endif()

add_subdirectory(mutrigana-base)
