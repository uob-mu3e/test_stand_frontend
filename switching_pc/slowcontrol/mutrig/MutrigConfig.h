#ifndef MUTRIG_CONFIG_H
#define MUTRIG_CONFIG_H

#include <string>
#include <tuple>
#include <map>
#include <vector>
#include <sstream>
#include "asic_config_base.h"
#include "mutrig_MIDAS_config.h"
#include "odbxx.h"
#include "midas.h"  //for return types

using midas::odb;

namespace mutrig {

class MutrigConfig:public mudaq::ASICConfigBase {
public:
    MutrigConfig();
    ~MutrigConfig();

    /**
     * Functions to parse MIDAS structs to MuTRiG patterns
     */
    void Parse_GLOBAL_from_struct(odb o);
    void Parse_TDC_from_struct(odb o);
    void Parse_CH_from_struct(odb o, int channel);
    void setParameterODBpp(std::string paraName, odb o);
    void setParameterODBpp(std::string paraName, odb o, int channel);

private:
    static paras_t parameters_ch;                             ///< static which stores the parameters for each channel (name, nbits, endian)
    static paras_t parameters_tdc;                            ///< static which stores the parameters for the tdcs (name, nbits, endian)
    static paras_t parameters_header;                         ///< static which stores the parameters for the header (name, nbits, endian)
    static paras_t parameters_footer;                         ///< static which stores the parameters for the footer (name, nbits, endian)
    static const unsigned int nch = 32;                       ///< number of channels used to generate config map

};

}// namespace mutrig

#endif // MUTRIG_CONFIG_H
