cmake_minimum_required(VERSION 3.1)

file(GLOB SOURCES "*.cpp")
file(GLOB INC "*.h")

add_executable(env_control ${SOURCES} ${PROJECT_SOURCE_DIR}/modules/midas/drivers/class/multi.cxx ${PROJECT_SOURCE_DIR}/modules/midas/drivers/device/mscbdev.cxx ${PROJECT_SOURCE_DIR}/modules/midas/mscb/src/mscb.cxx ${PROJECT_SOURCE_DIR}/modules/midas/drivers/class/hv.cxx)
target_include_directories(env_control PRIVATE ${MIDAS_DIR}/include ${MXML_DIR} ${INC} ${PROJECT_SOURCE_DIR}/modules/midas/drivers/ ${PROJECT_SOURCE_DIR}/modules/midas/mscb/include)
target_link_libraries(env_control mfe midas rt dl pthread util ${Boost_LIBRARIES})


install(TARGETS env_control DESTINATION bin)
