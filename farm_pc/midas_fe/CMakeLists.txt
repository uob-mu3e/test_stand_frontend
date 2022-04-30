FILE(GLOB_RECURSE INC_ALL "*.h")

include_directories(${LIBMUDAQ_DIR} ${INCMUDAQ_DIR})

if(CUDA_FOUND)
    set(CMAKE_CUDA_ARCHITECTURES 75)
    cuda_add_executable(farm_fe_gpu farm_fe.cu ${INC_ALL})
    target_link_libraries(farm_fe_gpu libmudaq -lpthread)

    target_include_directories(farm_fe_gpu PRIVATE ${INCMUDAQ_DIR})
    target_link_libraries(farm_fe_gpu libmudaq dl)
    target_link_libraries(farm_fe_gpu mfe midas rt pthread util)
    
    install(TARGETS farm_fe_gpu DESTINATION bin)
endif(CUDA_FOUND)
