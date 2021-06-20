
if(NOT MIDAS_FOUND)
    message(STATUS "Midas not found, building and installing in modules/midas")
# Run cmake for midas (needed for include file
    execute_process(COMMAND ${CMAKE_COMMAND} 
            -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} 
            ${CMAKE_CURRENT_SOURCE_DIR}/midas
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/midas/build)
# Build and install subproject
    execute_process(COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_SOURCE_DIR}/midas/build --target install  --parallel 4)
# Because Konstantin does not want to include mscb in MidasConfig.cmake
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
