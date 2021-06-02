
if(NOT MIDAS_FOUND)
    add_subdirectory(midas)
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/midas)
    find_package(Midas REQUIRED)
    add_subdirectory(midas/mscb)
endif()
if(ROOT_FOUND)
    if(USE_ANALYZER)
        add_subdirectory(analyzer)
    else()
        message(WARNING "USE_ANALYZER not set, if you want the analyzer, please set this variable")
    endif(USE_ANALYZER) 
else()
    message("No ROOT found, analyzer will not be build")
endif()


