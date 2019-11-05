#include <cstring>
#include <iostream>
#include <iomanip>

#include "mupix_config.h"

namespace mupix {


/// MUPIX configuration

MupixConfig::paras_t MupixConfig::parameters_chipdacs = {
        make_param("vnd2c_scale",        1, 1),
        make_param("vnd2c_offset",       2, 1),
        make_param("latchbias",          12, 0)
    };

MupixConfig::paras_t MupixConfig::parameters_pixeldacs = {
        make_param("energy_c_en",       1, 1),
        make_param("energy_r_en",       1, 1),
        make_param("mask",              1, 1)
     };


MupixConfig::MupixConfig() {
    // populate name/offset map
    length_bits = 0;
    // header 
    for(const auto& para : parameters_chipdacs)
        addPara(para, "");
    // pixels
    unsigned int nrow = 8;
    unsigned int ncol = 99;
    for(unsigned int row = 0; row < nrow; ++row) 
        for(unsigned int col = 0; col < ncol; ++col) 
            for(const auto& para : parameters_pixeldacs )
                addPara(para, "_"+std::to_string(row)+"_"+std::to_string(row));

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
