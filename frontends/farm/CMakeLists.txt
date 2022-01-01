
FILE(GLOB_RECURSE INC_ALL "*.h")

cuda_add_executable(farm_fe farm_fe.cu ${INC_ALL})
target_link_libraries(farm_fe libmudaq -lutil)

install(TARGETS farm_fe
        DESTINATION bin)
