#ifndef MUTRIG_CONFIG_H
#define MUTRIG_CONFIG_H

#include <string>
#include <tuple>
#include <map>
#include <vector>
#include <sstream>
#include "mutrig_MIDAS_config.h"
#include "midas.h"  //for return types
namespace mutrig {

/**
 * STiC configrations for the mudaq environment.
 */

class Config {
public:
    Config();
    ~Config();

    size_t length_bits;      ///< length of bitpattern in bits
    size_t length;           ///< length of bitpattern in full bytes
    size_t length_32bits;    ///< length of bitpattern in 32bit words


    /**
     * set parameter in write bitpattern by name
     * return 0 if success
     */
    int setParameter(std::string name, uint32_t value);

    /**
     * get parameter from read bitpattern
     */
    uint32_t getParameter(std::string);

    /**
     * resets bitpattern
     * input r:
     */
    int reset(char o = 'b');

    /**
     * debug function to get the current pattern as string
     */
    std::string getPattern();
    //check if readback pattern matches written pattern
    //return SUCCESS when written and read back patterns match, FAILURE otherwise
    int VerifyReadbackPattern();
    //Get error message giving bitpattern verification error informations.
    //Valid after calling VerifyReadbackPattern()
    std::string GetVerificationError(){ return m_verification_error; };

    friend std::ostream& operator<<(std::ostream& os, const Config& config);

    uint8_t * bitpattern_r; ///< bitpattern to be written
    uint8_t * bitpattern_w; ///< readback bitpattern

    /**
     * Functions to parse MIDAS structs to MuTRiG patterns
     */
    void Parse_GLOBAL_from_struct(MUTRIG_GLOBAL&);
    void Parse_TDC_from_struct(MUTRIG_TDC&);
    void Parse_CH_from_struct(MUTRIG_CH&, int channel);

protected:
    typedef std::tuple<unsigned int, size_t, bool> para_offset_t;   ///< mutrig parameter offset/nbits/endianess pair (offset, endianess)
    typedef std::map<std::string, para_offset_t> paras_offset_t;    ///< mutrig parameters offset/nbits/endianess map
    paras_offset_t paras_offsets;                                   ///< initialized from parameters on construction
    //error message giving bitpattern verification error informations. Valid after calling VerifyReadbackPattern()
    std::string m_verification_error;

public:
    /**
     * MuTRiG parameters (name, number of bits, endianess)
     */
    typedef std::tuple<std::string, size_t, bool> para_t;     ///< mutrig parameter (name, number of bits, endianess)
    typedef std::vector<para_t> paras_t;                      ///< mutrig parameter in correct order (starting at bit offset 0)

    const paras_offset_t& getParameters() const { return paras_offsets; }

private:
    static paras_t parameters_ch;                             ///< static which stores the parameters for each channel (name, nbits, endian)
    static paras_t parameters_tdc;                            ///< static which stores the parameters for the tdcs (name, nbits, endian)
    static paras_t parameters_header;                         ///< static which stores the parameters for the header (name, nbits, endian)
    static paras_t parameters_footer;                         ///< static which stores the parameters for the footer (name, nbits, endian)
    static const unsigned int nch = 32;                       ///< number of channels used to generate config map


    void addPara(const para_t&, const std::string);
};

}// namespace mutrig

#endif // MUTRIG_CONFIG_H
