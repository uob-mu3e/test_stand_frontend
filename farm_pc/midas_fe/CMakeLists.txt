FILE(GLOB_RECURSE INC_ALL "*.h")

include_directories(${LIBMUDAQ_DIR} ${INCMUDAQ_DIR})

if(CUDA_FOUND)
    set(CMAKE_CUDA_ARCHITECTURES 75)
    cuda_add_executable(farm_fe_gpu farm_fe.cu ${INC_ALL})
    target_link_libraries(farm_fe_gpu libmudaq -lpthread)

    target_include_directories(farm_fe_gpu PRIVATE ${INCMUDAQ_DIR})
    target_link_libraries(farm_fe_gpu libmudaq dl)
    target_link_libraries(farm_fe_gpu mfe midas rt pthread util)
endif(CUDA_FOUND)

add_executable(ro_swb_fe ro_swb_fe.cpp ${INC_ALL})
target_link_libraries(ro_swb_fe libmudaq -lpthread)

target_include_directories(ro_swb_fe PRIVATE ${INCMUDAQ_DIR})
target_link_libraries(ro_swb_fe libmudaq dl)
target_link_libraries(ro_swb_fe mfe midas rt pthread util)

if(CUDA_FOUND)
    install(TARGETS ro_swb_fe farm_fe_gpu DESTINATION bin)
else()
    install(TARGETS ro_swb_fe DESTINATION bin)
endif(CUDA_FOUND)
