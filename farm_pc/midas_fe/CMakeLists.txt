FILE(GLOB_RECURSE INC_ALL "*.h")

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -std=c++11
)

include_directories(${LIBMUDAQ_DIR} ${INCMUDAQ_DIR})

cuda_add_executable(farm_fe farm_fe.cu ${INC_ALL})
message(${INCMUDAQ_DIR})
target_include_directories(farm_fe PRIVATE ${INCMUDAQ_DIR})
target_link_libraries(farm_fe -lcuda libmudaq -lutil)
target_link_libraries(farm_fe mfe midas rt pthread util)

install(TARGETS farm_fe
        DESTINATION bin)
