FILE(GLOB_RECURSE INC_ALL "*.h")

cuda_add_executable(farm_fe farm_fe.cu ${INC_ALL})
target_link_libraries(farm_fe -lcuda libmudaq -lutil)
target_link_libraries(farm_fe ${MIDASLIB} ${MIDASLIBDIR}/mfe.o)
target_include_directories(farm_fe PRIVATE ${MIDASINCLUDEDIR} ${LIBMUDAQDIR})

install(TARGETS farm_fe
        DESTINATION bin)
