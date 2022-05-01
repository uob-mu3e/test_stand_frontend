cmake_minimum_required(VERSION 3.1)

project(farm_pc_tools CXX)

if(CUDA_FOUND)
    set(CMAKE_CUDA_ARCHITECTURES 75)
    cuda_add_executable(gpu_dmatest gpu_dmatest.cu ${INC_ALL})
    target_link_libraries(gpu_dmatest libmudaq -lpthread)
    target_include_directories(gpu_dmatest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/include)
endif(CUDA_FOUND)

include_directories(${LIBMUDAQ_DIR})
add_executable(rw rw.cpp ${INC_ALL})
add_executable(ddr3_test ddr3_test.cpp ${INC_ALL})
add_executable(swb_dmatest swb_dmatest.cpp ${INC_ALL})
add_executable(multithread_test multithread_test.cpp ${INC_ALL})
add_executable(reset_link_a10_test reset_link_a10_test.cpp ${INC_ALL})
target_link_libraries(rw libmudaq -lutil)
target_link_libraries(ddr3_test libmudaq -lutil)
target_link_libraries(swb_dmatest libmudaq -lutil)
target_link_libraries(multithread_test -lpthread -lutil)
target_link_libraries(reset_link_a10_test libmudaq -lutil)
target_include_directories(test_slowcontrol PRIVATE ${LIBSLOWCONTROL_DIR})
target_include_directories(swb_dmatest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../common/firmware/include)

if(CUDA_FOUND)
    install(TARGETS rw ddr3_test swb_dmatest gpu_dmatest reset_link_a10_test
            DESTINATION bin)
else()
    install(TARGETS rw ddr3_test swb_dmatest reset_link_a10_test
            DESTINATION bin)
endif(CUDA_FOUND)
