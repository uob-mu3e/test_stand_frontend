/********************************************************************\

  Name:         mupix_config.h
  Created by:   Konrad Briggl

  Contents:     Assembly of bitpatterns from ODB
                Configuration class MupixConfig contains bitpattern for one row.
		Row address parameter is set during construction or by SetRowAddress().
		The configuration is then assembled as follows:
		rowaddr | chipdacs_g1 | pixelrowdacs | chipdacs_g2

  Created on:   Nov 05 2019

\********************************************************************/

#ifndef MUPIX_CONFIG_H
#define MUPIX_CONFIG_H
#include "midas.h"
#include "asic_config_base.h"
#include "mupix_MIDAS_config.h"

namespace mupix{

class MupixConfig: public mudaq::ASICConfigBase{
public:
    MupixConfig();
    ~MupixConfig();
    //Set row selection bits
    void SetRow(int16_t row); 
    void Parse_ChipDACs_from_struct(MUPIX_CHIPDACS mt);
    void Parse_PixelRowDACs_from_struct(MUPIX_PIXELROWCONFIG mt);

private:
    static paras_t parameters_chipdacs;                             ///< parameters for global dacs (name, nbits, endian) 
    static paras_t parameters_pixeldacs;                             ///< parameters for each pixel

};

class MupixBoardConfig: public mudaq::ASICConfigBase{
public:
    MupixBoardConfig();
    ~MupixBoardConfig();
    
    void Parse_BoardDACs_from_struct(MUPIX_BOARDDACS mt);
private:
    static paras_t parameters_boarddacs;                             ///< parameters for global dacs (name, nbits, endian) 
};

}
#endif //MUPIX_CONFIG_H
