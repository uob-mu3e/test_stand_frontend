FILE(GLOB_RECURSE INC_ALL "*.h")

include_directories(${LIBMUDAQ_DIR} ${INCMUDAQ_DIR})

find_package(CUDA)
if(CUDA_FOUND)
    set(CMAKE_CUDA_ARCHITECTURES 75)
    cuda_add_executable(farm_fe farm_fe.cu ${INC_ALL})
    target_link_libraries(farm_fe libmudaq -lpthread)
endif(CUDA_FOUND)

add_executable(gpu_fe gpu_fe.cpp ${INC_ALL})
message(${INCMUDAQ_DIR})
target_include_directories(farm_fe PRIVATE ${INCMUDAQ_DIR})
target_link_libraries(farm_fe libmudaq dl)
target_link_libraries(farm_fe mfe midas rt pthread util)
target_include_directories(gpu_fe PRIVATE ${INCMUDAQ_DIR})
target_link_libraries(gpu_fe libmudaq dl)
target_link_libraries(gpu_fe mfe midas rt pthread util)

if(CUDA_FOUND)
    install(TARGETS farm_fe gpu_fe
            DESTINATION bin)
else()
    install(TARGETS gpu_fe
            DESTINATION bin)
endif(CUDA_FOUND)
