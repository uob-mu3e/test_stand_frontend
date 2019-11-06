#include <cstring>
#include <iostream>
#include <iomanip>

#include "mupix_config.h"

namespace mupix {


/// MUPIX configuration
// Pattern is assembled based on the scheme described in the mupix8 documentation, section 10

MupixConfig::paras_t MupixConfig::parameters_chipdacs = {
        make_param("vnd2c_scale",        1, 1),
        make_param("vnd2c_offset",       2, 1),
        make_param("latchbias",          12, 0)
    };

MupixConfig::paras_t MupixConfig::parameters_pixeldacs = {
        make_param("RAM_IN",            6, {5,4,3,0,1,2}),
        make_param("enable_hitbus",     1, 0),
        make_param("enable_injection",  1, 0)
     };


MupixConfig::MupixConfig() {
    unsigned int nrow = 200;
    unsigned int ncol = 128;
    // populate name/offset map
    length_bits = 0;
    // MUPIX DAC block 1 (10.1)
    for(const auto& para : parameters_chipdacs)
        addPara(para, "");
    // row selection register (write bits, 10.2)
    addPara(make_param("rowsel_reg",nrow,0), "");
    // pixel row block (pixel settings, 10.3).
    for(unsigned int col = 0; col < ncol; ++col) 
        for(const auto& para : parameters_pixeldacs )
            addPara(para, "_col"+std::to_string(col));
    // row register (Q bits, 10.4 )
    --row for(unsigned int col = 0; col < ncol; ++col) 
    for(const auto& para : parameters_rowdacs)
        addPara(para, "");
    // MUPIX DAC block 2 (10.5)
    for(const auto& para : parameters_chipdacs2)
        addPara(para, "");
    for(const auto& para : parameters_voltagedacs)
        addPara(para, "");



    // allocate memory for bitpattern
    length = length_bits/8;
    if( length_bits%8 > 0 ) length++;
    length_32bits = length/4;
    if( length%4 > 0 ) length_32bits++;
    bitpattern_r = new uint8_t[length_32bits*4]; 
    bitpattern_w = new uint8_t[length_32bits*4]; 
    reset();
}

MupixConfig::~MupixConfig() {
    delete[] bitpattern_r;
    delete[] bitpattern_w;
}

void MupixConfig::Parse_ChipDACs_from_struct(MUPIX_CHIPDACS mt_g){
    //setParameter("ext_trig_mode", mt_g.ext_trig_mode);
    //setParameter("ext_trig_endtime_sign", mt_g.ext_trig_endtime_sign);
}

/// MUPIX board configuration

MupixBoardConfig::paras_t MupixBoardConfig::parameters_boarddacs = {
        make_param("vnd2c_scale",        1, 1),
        make_param("vnd2c_offset",       2, 1),
        make_param("latchbias",          12, 0)
    };



MupixBoardConfig::MupixBoardConfig() {
    // populate name/offset map
    length_bits = 0;
    // header 
    for(const auto& para : parameters_boarddacs)
        addPara(para, "");

    // allocate memory for bitpattern
    length = length_bits/8;
    if( length_bits%8 > 0 ) length++;
    length_32bits = length/4;
    if( length%4 > 0 ) length_32bits++;
    bitpattern_r = new uint8_t[length_32bits*4]; 
    bitpattern_w = new uint8_t[length_32bits*4]; 
    reset();	
}

MupixBoardConfig::~MupixBoardConfig() {
    delete[] bitpattern_r;
    delete[] bitpattern_w;
}

void MupixBoardConfig::Parse_BoardDACs_from_struct(MUPIX_BOARDDACS mt_g){
    //setParameter("ext_trig_mode", mt_g.ext_trig_mode);
    //setParameter("ext_trig_endtime_sign", mt_g.ext_trig_endtime_sign);
}


} // namespace mutrig
