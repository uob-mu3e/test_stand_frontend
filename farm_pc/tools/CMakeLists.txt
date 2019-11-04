FILE(GLOB_RECURSE INC_ALL "*.h")

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -std=c++11
)

include_directories(${MIDASINCLUDE_DIR} ${LIBMUDAQ_DIR})
add_executable(rw rw.cpp ${INC_ALL})
cuda_add_executable(dmatest dmatest.cu ${INC_ALL})
cuda_add_executable(dma_speed_test dma_speed_test.cu ${INC_ALL})
target_link_libraries(rw libmudaq -lutil)
target_link_libraries(dmatest -lcuda libmudaq -lutil)
target_link_libraries(dma_speed_test -lcuda libmudaq -lutil)

message(STATUS "Minclude ${MIDASINCLUDE_DIR}")