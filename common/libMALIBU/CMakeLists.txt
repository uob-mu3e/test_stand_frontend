add_library(libmalibu STATIC
    include/MALIBU.h
    src/MALIBU.cpp
)
target_link_libraries(libmalibu  libmudaq ${Boost_LIBRARIES})
target_include_directories(libmalibu
PUBLIC 
	"include"
PRIVATE
	${LIBMUDAQ_DIR}
)
