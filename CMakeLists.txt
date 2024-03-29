cmake_minimum_required(VERSION 3.1)

project(mu3e_online LANGUAGES CXX)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "" FORCE)
    set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT FALSE)
endif()
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# MU3E_ONLINE_GIT_REF
#find_package(Git 1.8 REQUIRED)
#execute_process(
#    COMMAND ${GIT_EXECUTABLE} describe --abbrev=16 --dirty
#    OUTPUT_STRIP_TRAILING_WHITESPACE
#    OUTPUT_VARIABLE MU3E_ONLINE_GIT_REF
#)
#message(STATUS "MU3E_ONLINE_GIT_REF: ${MU3E_ONLINE_GIT_REF}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(
    "-Wpedantic" "-Wall" "-Wextra" "-Wno-missing-field-initializers"
    "-Wformat=2"
)

set(Boost_NO_SYSTEM_PATHS TRUE)
if (Boost_NO_SYSTEM_PATHS)
    if(DEFINED ENV{CVMFS_ROOT})
        set(BOOST_ROOT "$ENV{CVMFS_ROOT}/sft.cern.ch/lcg/releases/LCG_102/Boost/1.78.0/x86_64-centos7-gcc11-opt")
    endif()
    set(BOOST_INCLUDE_DIRS "${BOOST_ROOT}/include")
    set(BOOST_LIBRARY_DIRS "${BOOST_ROOT}/lib")
endif (Boost_NO_SYSTEM_PATHS)
include_directories(${BOOST_INCLUDE_DIRS})

# required packages
find_package(Boost 1.66 REQUIRED COMPONENTS thread system program_options)
find_package(Git 1.8 REQUIRED)
find_package(ROOT REQUIRED COMPONENTS
    RIO Hist Tree Net
)

# optional packages
find_package(Doxygen 1.8)
message("here we set GLVND to default (https://cmake.org/cmake/help/git-stage/policy/CMP0072.html)")
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL)
find_package(CUDA)

set(LIBMUDAQ_DIR ${PROJECT_SOURCE_DIR}/common/libmudaq)
set(LIBMUPIX_DIR ${PROJECT_SOURCE_DIR}/common/libmupix)
set(LIBMUTRIG_DIR ${PROJECT_SOURCE_DIR}/common/libmutrig)
set(LIBMALIBU_DIR ${PROJECT_SOURCE_DIR}/common/libMALIBU)
set(INCMUDAQ_DIR ${PROJECT_SOURCE_DIR}/common/include)

# on some older boxes, wpedantic and MIDAS do not mix
add_subdirectory(modules)
find_package(Midas REQUIRED
    HINTS
        $ENV{MIDASSYS}
        ~/midas
        ~/online/modules/midas
        ${CMAKE_CURRENT_SOURCE_DIR}/../midas
        )
# To be implemented in midas/MidasConfig.cmake at some point
if(NOT EXISTS ${MIDAS_LIBRARY_DIRS})
    message(FATAL_ERROR "MIDAS not found.
    Please add your instance of midas to \"cmake -DCMAKE_PREFIX_PATH=PATHTOMIDAS ..\".
    Also make sure that you did make install after installing midas.
    If you have, consider deleting your build directory")
endif()
get_filename_component(MIDASSYS ${MIDAS_LIBRARY_DIRS} DIRECTORY)
message(STATUS "MIDASSYS = ${MIDASSYS}")

list (APPEND MIDAS_INCLUDE_DIRS
    ${MIDASSYS}/midas/manalyzer
    ${MIDASSYS}/midas/midasio
    ${MIDASSYS}/midas/mvodb
    )
link_directories(${MIDAS_LIBRARY_DIRS})
include_directories(${MIDAS_INCLUDE_DIRS})

add_subdirectory(common)
add_subdirectory(backend_pc)
add_subdirectory(switching_pc)

if(APPLE)
    message("Compile for Apple")
    add_compile_definitions(OS_DARWIN)
    # remove rt libary from crfe, netfe, copyHostnames, power_supply_fe, switch_fe, etc.
    # TODO: maybe this can be done more nicely
    macro(rm_rt TARGETNAME)
        get_target_property(TARGET_LIBRARIES_RT ${TARGETNAME} LINK_LIBRARIES)
        LIST(REMOVE_ITEM TARGET_LIBRARIES_RT rt)
        set_property(TARGET ${TARGETNAME} PROPERTY LINK_LIBRARIES ${TARGET_LIBRARIES_RT})
    endmacro()

    rm_rt(crfe)
    rm_rt(netfe)
    rm_rt(copyHostnames)
    rm_rt(power_supply_fe)
    rm_rt(MpTuneDACsDB)
    rm_rt(usb_clock_pi)
    rm_rt(power)
    rm_rt(switch_fe)
endif(APPLE)

add_subdirectory(farm_pc/midas_fe)
add_subdirectory(farm_pc/tools)

if(CMAKE_PROJECT_NAME STREQUAL mu3e_online)
    enable_testing()
    include(GoogleTest)
    add_custom_target(build_and_test ${CMAKE_CTEST_COMMAND} -V)
    add_subdirectory(common/tests EXCLUDE_FROM_ALL)
endif()

configure_file(set_env.sh.in set_env.sh @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/set_env.sh DESTINATION ./)

configure_file(make_links.sh.in make_links.sh @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/make_links.sh DESTINATION ./)

configure_file(setup_odb.sh.in setup_odb.sh @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/setup_odb.sh DESTINATION ./)

configure_file(setup_feb.sh.in setup_feb.sh @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/setup_feb.sh DESTINATION ./)

configure_file(start_daq.sh.in start_daq.sh @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/start_daq.sh DESTINATION bin
PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

configure_file(initiate_power_supply.sh.in initiate_power_supply.sh @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/initiate_power_supply.sh DESTINATION ./)

configure_file(arduino_interface.sh.in arduino_interface.sh @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/arduino_interface.sh DESTINATION ./)

# teststand frontend in online
add_subdirectory(online)
