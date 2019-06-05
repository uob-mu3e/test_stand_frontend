include_directories(${MIDASINCLUDEDIR})

FILE(GLOB_RECURSE INC_ALL "*.h")

cuda_add_executable(switching_fe switching_fe.cu ${INC_ALL})
target_link_libraries(switching_fe -lcuda libmudaq -lutil)
target_link_libraries(switching_fe ${MIDASLIB} ${MIDASLIBDIR}/mfe.o)

install(TARGETS switching_fe
        DESTINATION bin)
