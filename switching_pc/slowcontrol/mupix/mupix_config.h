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

class odb;

namespace mupix{

class MupixConfig: public mudaq::ASICConfigBase{
public:
    MupixConfig();
    ~MupixConfig();
    void Parse_BiasDACs_from_struct(MUPIX_BIASDACS mt);
    void Parse_ConfDACs_from_struct(MUPIX_CONFDACS mt);
    void Parse_VDACs_from_struct(MUPIX_VDACS mt);

    void Parse_BiasDACs_from_odb(odb & mt);
    void Parse_ConfDACs_from_odb(odb & mt);
    void Parse_VDACs_from_odb(odb & mt);
private:
    static paras_t parameters_bias;                             ///< parameters for bias dacs
    static paras_t parameters_conf;                             ///< parameters for conf dacs
    static paras_t parameters_vdacs;                             ///< parameters for vdac dacs
};

}
#endif //MUPIX_CONFIG_H
