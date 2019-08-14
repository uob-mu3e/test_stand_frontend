FILE(GLOB_RECURSE INC_ALL "*.h")

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -std=c++11
)

include_directories(${MIDASINCLUDE_DIR} ${LIBMUDAQ_DIR})
cuda_add_executable(farm_fe farm_fe.cu ${INC_ALL})
cuda_add_executable(dmatest dmatest.cu ${INC_ALL})
target_link_libraries(farm_fe -lcuda libmudaq -lutil)
target_link_libraries(dmatest -lcuda libmudaq -lutil)
target_link_libraries(farm_fe mfe midas rt pthread util)

message(STATUS "Minclude ${MIDASINCLUDE_DIR}")

install(TARGETS farm_fe
        DESTINATION bin)
