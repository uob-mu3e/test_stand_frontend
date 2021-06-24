if(NOT MIDAS_FOUND)
    message(STATUS "Midas not found, building and installing in modules/midas")
    file(MAKE_DIRECTORY midas/build)
# Run cmake for midas (needed for include file
    execute_process(COMMAND ${CMAKE_COMMAND} 
            -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} 
            ${CMAKE_CURRENT_SOURCE_DIR}/midas
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/midas/build)
# Build and install subproject
    execute_process(COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_SOURCE_DIR}/midas/build --target install  --parallel 4)
    # At the moment, mscb is not fully part of the midas cmake config
    add_subdirectory(midas/mscb)

    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/midas")
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} PARENT_SCOPE)
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
