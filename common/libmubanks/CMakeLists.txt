add_library(libmubanks STATIC
    mu3ebanks.cpp
    )
target_include_directories(libmubanks PUBLIC .)
target_include_directories(libmubanks PRIVATE ../include)
target_link_libraries(libmubanks midas)
