FILE(GLOB_RECURSE INC_ALL "*.h")

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -std=c++17
)

cuda_add_executable(gpu_dmatest gpu_dmatest.cu ${INC_ALL})
target_link_libraries(gpu_dmatest -lcuda libmudaq -lutil)

include_directories(${LIBMUDAQ_DIR})
add_executable(rw rw.cpp ${INC_ALL})
add_executable(ddr3_test ddr3_test.cpp ${INC_ALL})
add_executable(swb_dmatest swb_dmatest.cpp ${INC_ALL})
target_link_libraries(rw libmudaq -lutil)
target_link_libraries(ddr3_test libmudaq -lutil)
target_link_libraries(swb_dmatest libmudaq -lutil)

install(TARGETS rw ddr3_test swb_dmatest gpu_dmatest
        DESTINATION bin)
