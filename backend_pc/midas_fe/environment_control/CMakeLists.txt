add_executable(sc_fe_sensinmagnet 
   sc_fe_sensinmagnet.cxx
   ${MIDASSYS}/src/mfe.cxx
   ${MIDASSYS}/drivers/class/multi.cxx
   ${MIDASSYS}/drivers/device/mscbdev.cxx
   ${MIDASSYS}/mscb/src/mscb.cxx)

target_include_directories(sc_fe_sensinmagnet PRIVATE
   ${MIDASSYS}/include 
   ${MIDASSYS}/drivers
   ${MIDASSYS}/mxml
   ${MIDASSYS}/mscb/include)

target_link_libraries(sc_fe_sensinmagnet midas)

# add required linux libraries
if (${CMAKE_SYSTEM_NAME} MATCHES Linux)
   target_link_libraries(sc_fe_sensinmagnet
      pthread
      util
      rt
      dl) 
endif() 

install(TARGETS sc_fe_sensinmagnet DESTINATION bin)
