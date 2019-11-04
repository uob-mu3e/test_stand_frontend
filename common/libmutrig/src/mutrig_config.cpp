#include <cstring>
#include <iostream>
#include <iomanip>

#include "mutrig_config.h"

namespace mutrig {

Config::paras_t Config::parameters_tdc = {
        std::make_tuple("vnd2c_scale",        1, 1),
        std::make_tuple("vnd2c_offset",       2, 1),
        std::make_tuple("vnd2c",              6, 1),
        std::make_tuple("vncntbuffer_scale",  1, 1),
        std::make_tuple("vncntbuffer_offset", 2, 1),
        std::make_tuple("vncntbuffer",        6, 1),
        std::make_tuple("vncnt_scale",        1, 1),
        std::make_tuple("vncnt_offset",       2, 1),
        std::make_tuple("vncnt",              6, 1),
        std::make_tuple("vnpcp_scale",        1, 1),
        std::make_tuple("vnpcp_offset",       2, 1),
        std::make_tuple("vnpcp",              6, 1),
        std::make_tuple("vnvcodelay_scale",   1, 1),
        std::make_tuple("vnvcodelay_offset",  2, 1),
        std::make_tuple("vnvcodelay",         6, 1),
        std::make_tuple("vnvcobuffer_scale",  1, 1),
        std::make_tuple("vnvcobuffer_offset", 2, 1),
        std::make_tuple("vnvcobuffer",        6, 1),
        std::make_tuple("vnhitlogic_scale",   1, 1),
        std::make_tuple("vnhitlogic_offset",  2, 1),
        std::make_tuple("vnhitlogic",         6, 1),
        std::make_tuple("vnpfc_scale",        1, 1),
        std::make_tuple("vnpfc_offset",       2, 1),
        std::make_tuple("vnpfc",              6, 1),
        std::make_tuple("latchbias",          12, 0)
    };

Config::paras_t Config::parameters_ch = {
        std::make_tuple("energy_c_en",       1, 1), //old name: anode_flag
        std::make_tuple("energy_r_en",       1, 1), //old name: cathode_flag
        std::make_tuple("sswitch",           1, 1),
        std::make_tuple("cm_sensing_high_r", 1, 1), //old name: SorD; should be always '0'
        std::make_tuple("amon_en_n",         1, 1), //old name: SorD_not; 0: enable amon in the channel
        std::make_tuple("edge",              1, 1),
        std::make_tuple("edge_cml",          1, 1),
        std::make_tuple("cml_sc",            1, 1),
        std::make_tuple("dmon_en",           1, 1),
        std::make_tuple("dmon_sw",           1, 1),
        std::make_tuple("tdctest_n",           1, 1),
        std::make_tuple("amonctrl",          3, 1),
        std::make_tuple("comp_spi",          2, 1),
        std::make_tuple("sipm_sc",           1, 1),
        std::make_tuple("sipm",              6, 1),
        std::make_tuple("tthresh_sc",        3, 1),
        std::make_tuple("tthresh",           6, 1),
        std::make_tuple("ampcom_sc",         2, 1),
        std::make_tuple("ampcom",            6, 1),
        std::make_tuple("inputbias_sc",      1, 1),
        std::make_tuple("inputbias",         6, 1),
        std::make_tuple("ethresh",           8, 1),
        std::make_tuple("pole_sc",           1, 1),
        std::make_tuple("pole",              6, 1),
        std::make_tuple("cml",               4, 1),
        std::make_tuple("delay",             1, 1),
        std::make_tuple("pole_en_n",         1, 1), //old name: dac_delay_bit1; 0: DAC_pole on
        std::make_tuple("mask",              1, 1)
     };

Config::paras_t Config::parameters_header = {
        std::make_tuple("gen_idle",              1, 1),
        std::make_tuple("recv_all",              1, 1),
        std::make_tuple("ext_trig_mode",         1, 1), // new 
        std::make_tuple("ext_trig_endtime_sign", 1, 1), // sign of the external trigger matching window, 1: end time is after the trigger; 0: end time is before the trigger
        std::make_tuple("ext_trig_offset",       4, 0), // offset of the external trigger matching window
        std::make_tuple("ext_trig_endtime",      4, 0), // end time of external trigger matching window
        std::make_tuple("ms_limits",             5, 0),
        std::make_tuple("ms_switch_sel",         1, 1),
        std::make_tuple("ms_debug",              1, 1),
        std::make_tuple("prbs_debug",            1, 1), // new
        std::make_tuple("prbs_single",           1, 1), // new
        std::make_tuple("short_event_mode",      1, 1), //fast transmission mode
        std::make_tuple("pll_setcoarse",         1, 1),
        std::make_tuple("pll_envomonitor",       1, 1),
        std::make_tuple("disable_coarse",        1, 1)
    };

Config::paras_t Config::parameters_footer = {
        std::make_tuple("amon_en",       1, 1),
        std::make_tuple("amon_dac",      8, 1),
        std::make_tuple("dmon_1_en",     1, 1),
        std::make_tuple("dmon_1_dac",    8, 1),
        std::make_tuple("dmon_2_en",     1, 1),
        std::make_tuple("dmon_2_dac",    8, 1),
        std::make_tuple("lvds_tx_vcm",   8, 1), // new
        std::make_tuple("lvds_tx_bias",  6, 1)  // new
    };

void Config::addPara(const para_t& para, const std::string postfix) {
    paras_offsets[std::get<0>(para)+postfix] = std::make_tuple(length_bits, std::get<1>(para), std::get<2>(para));
    length_bits += std::get<1>(para);
}



Config::Config() {
    // populate name/offset map

    length_bits = 0;
    // header 
    for(const auto& para : parameters_header )
        addPara(para, "");
    for(unsigned int ch = 0; ch < nch; ++ch) { 
        for(const auto& para : parameters_ch )
            addPara(para, "_"+std::to_string(ch));
    }
    for(const auto& para : parameters_tdc )
        addPara(para, "");
    //for(const auto& para : parameters_tdc ) addPara(para, "_right");
    for(const auto& para : parameters_footer )
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

Config::~Config() {
    delete[] bitpattern_r;
    delete[] bitpattern_w;
}

int Config::setParameter(std::string name, uint32_t value) {
    auto para = paras_offsets.find(name);
    if( para == paras_offsets.end() ) {
        std::cerr << "Parameter '" << name << "' is not present in mutrig config" << std::endl;
        return 1; // parameter name not present
    }
    unsigned int offset;
    size_t nbits;
    bool endianess;
    std::tie(offset, nbits, endianess) = para->second;
    if( (value >> nbits) != 0 ) {
        std::cerr << "Value '" << value << "' outside of range of " << nbits << " bits." << std::endl;
        return 2; // out of range
    }

    uint32_t mask = 0x01;
    unsigned int pos = offset + (endianess == false ? 0 : (nbits - 1) );
    for(; (pos < offset + nbits) && (pos >= offset);
        pos = pos + (endianess == false ? +1 : -1 ), mask <<= 1) {
        unsigned int n = pos%8;
        //unsigned int x = (mask & value) != 0 
        if ((mask & value) != 0 ) bitpattern_w[pos/8] |=   1 << n;  // set nth bit 
        else                      bitpattern_w[pos/8] &= ~(1 << n); // clear nth bit
        //bitpattern_w[pos/8] ^= (-x ^ bitpattern_w[pos/8]) & (1 << n); // set nth bit to x 
    }
    return 0;
}

uint32_t Config::getParameter(std::string name) {
    auto para = paras_offsets.find(name);
    if( para == paras_offsets.end() ) {
        std::cerr << "Parameter '" << name << "' is not present in mutrig config" << std::endl;
        return uint32_t(-1); // parameter name not present
    }
    unsigned int offset;
    size_t nbits;
    bool endianess;
    std::tie(offset, nbits, endianess) = para->second;

    uint32_t value = 0;
    unsigned int value_bit = 0;
    unsigned int pos = offset + (endianess == false ? 0 : (nbits - 1) );
    for(; (pos < offset + nbits) && (pos >= offset);
        pos = pos + (endianess == false ? +1 : -1 ), value_bit++) {
        value += ( ( (bitpattern_r[pos/8] & (1 << pos%8)) != 0 ) << value_bit);
    }
    return value;
}

int Config::reset(char o) {
    switch(o) {
      case 'r': std::memset(bitpattern_r, 0, length_32bits*4);
                break;
      case 'w': std::memset(bitpattern_w, 0, length_32bits*4);
                break;
      default:  std::memset(bitpattern_r, 0, length_32bits*4);
                std::memset(bitpattern_w, 0, length_32bits*4);
    }
    return 0;
}

std::string Config::getPattern() {
    std::stringstream buffer;
    buffer << *this;
    return buffer.str();
}

int Config::VerifyReadbackPattern(){
    m_verification_error="";
    for(size_t i=0;i<length-1;++i){
    	if(bitpattern_w[i]!=bitpattern_r[i]){
		m_verification_error="Write/Read mismatch at byte "+std::to_string(i)+", written value "+std::to_string(bitpattern_w[i])+" vs. "+std::to_string(bitpattern_r[i]);
		return FE_ERR_HW;
	}
    }
    for(size_t i=0;i<length_bits%8;i++){
    	if((bitpattern_w[length-1]&(1<<i))!=(bitpattern_r[length-1]&(1<<i))){
		m_verification_error="Write/Read mismatch at last byte ("+std::to_string(i)+"), written value "+std::to_string(bitpattern_w[i])+" vs. "+std::to_string(bitpattern_r[i]);
		return FE_ERR_HW;
	}
    
    }
    return FE_SUCCESS;
}

std::ostream& operator<<(std::ostream& os, const Config& config) {
    os << " bitpattern: (" << config.length << "/" << config.length_bits << ") 0x" << std::hex;
    os << std::endl<<" write: 0x" << std::hex;
    for( unsigned int i = 0; i < config.length; i++) os << std::setw(2) << std::setfill('0') << ((uint16_t)config.bitpattern_w[config.length - i - 1]); // << " ";// << config.bitpattern_r[i] << " ";
    //os << std::endl;
    os << std::endl<<" read:  0x" << std::hex;
    for( unsigned int i = 0; i < config.length; i++) os << std::setw(2) << std::setfill('0') << ((uint16_t)config.bitpattern_r[config.length - i - 1]);
    os << std::endl;
    os << std::dec;
    //std::cout << "bitpattern write: 0x" << std::hex << std::setw(2) << std::setfill('0') << config.bitpattern_w[0] << std::endl; 
    return os;
}

void Config::Parse_GLOBAL_from_struct(MUTRIG_GLOBAL& mt_g){
    //hard coded in order to avoid macro magic
//    setParameter("", mt_g.n_asics);
//    setParameter("", mt_g.n_channels);
    setParameter("ext_trig_mode", mt_g.ext_trig_mode);
    setParameter("ext_trig_endtime_sign", mt_g.ext_trig_endtime_sign);
    setParameter("ext_trig_offset", mt_g.ext_trig_offset);
    setParameter("ext_trig_endtime", mt_g.ext_trig_endtime);
    setParameter("gen_idle", mt_g.gen_idle);
    setParameter("ms_debug", mt_g.ms_debug);
    setParameter("prbs_debug", mt_g.prbs_debug);
    setParameter("prbs_single", mt_g.prbs_single);
    setParameter("recv_all", mt_g.recv_all);
    setParameter("disable_coarse", mt_g.disable_coarse);
    setParameter("pll_setcoarse", mt_g.pll_setcoarse);
    setParameter("short_event_mode", mt_g.short_event_mode);
    setParameter("pll_envomonitor", mt_g.pll_envomonitor);
    setParameter("lvds_tx_vcm", mt_g.lvds_tx_vcm);
    setParameter("lvds_tx_bias", mt_g.lvds_tx_bias);
}

void Config::Parse_TDC_from_struct(MUTRIG_TDC& mt_tdc){
    setParameter("vnpfc", mt_tdc.vnpfc);
    setParameter("vnpfc_offset", mt_tdc.vnpfc_offset);
    setParameter("vnpfc_scale", mt_tdc.vnpfc_scale);
    setParameter("vncnt", mt_tdc.vncnt);
    setParameter("vncnt_offset", mt_tdc.vncnt_offset);
    setParameter("vncnt_scale", mt_tdc.vncnt_scale);
    setParameter("vnvcobuffer", mt_tdc.vnvcobuffer);
    setParameter("vnvcobuffer_offset", mt_tdc.vnvcobuffer_offset);
    setParameter("vnvcobuffer_scale", mt_tdc.vnvcobuffer_scale);
    setParameter("vnd2c", mt_tdc.vnd2c);
    setParameter("vnd2c_offset", mt_tdc.vnd2c_offset);
    setParameter("vnd2c_scale", mt_tdc.vnd2c_scale);
    setParameter("vnpcp", mt_tdc.vnpcp);
    setParameter("vnpcp_offset", mt_tdc.vnpcp_offset);
    setParameter("vnpcp_scale", mt_tdc.vnpcp_scale);
    setParameter("vnhitlogic", mt_tdc.vnhitlogic);
    setParameter("vnhitlogic_offset", mt_tdc.vnhitlogic_offset);
    setParameter("vnhitlogic_scale", mt_tdc.vnhitlogic_scale);
    setParameter("vncntbuffer", mt_tdc.vncntbuffer);
    setParameter("vncntbuffer_offset", mt_tdc.vncntbuffer_offset);
    setParameter("vncntbuffer_scale", mt_tdc.vncntbuffer_scale);
    setParameter("vnvcodelay", mt_tdc.vnvcodelay);
    setParameter("vnvcodelay_offset", mt_tdc.vnvcodelay_offset);
    setParameter("vnvcodelay_scale", mt_tdc.vnvcodelay_scale);
    setParameter("latchbias", mt_tdc.latchbias);
    setParameter("ms_limits", mt_tdc.ms_limits);
    setParameter("ms_switch_sel", mt_tdc.ms_switch_sel);
    setParameter("amon_en", mt_tdc.amon_en);
    setParameter("amon_dac", mt_tdc.amon_dac);
    setParameter("dmon_1_en", mt_tdc.dmon_1_en);
    setParameter("dmon_1_dac", mt_tdc.dmon_1_dac);
    setParameter("dmon_2_en", mt_tdc.dmon_2_en);
    setParameter("dmon_2_dac", mt_tdc.dmon_2_dac);

}


void Config::Parse_CH_from_struct(MUTRIG_CH& mt_ch, int channel){
    setParameter("mask_" + std::to_string(channel), mt_ch.mask);
    setParameter("tthresh_" + std::to_string(channel), mt_ch.tthresh);
    setParameter("tthresh_sc_" + std::to_string(channel), mt_ch.tthresh_sc);
    setParameter("ethresh_" + std::to_string(channel), mt_ch.ethresh);
    setParameter("sipm_" + std::to_string(channel), mt_ch.sipm);
    setParameter("sipm_sc_" + std::to_string(channel), mt_ch.sipm_sc);
    setParameter("inputbias_" + std::to_string(channel), mt_ch.inputbias);
    setParameter("inputbias_sc_" + std::to_string(channel), mt_ch.inputbias_sc);
    setParameter("pole_" + std::to_string(channel), mt_ch.pole);
    setParameter("pole_sc_" + std::to_string(channel), mt_ch.pole_sc);
    setParameter("ampcom_" + std::to_string(channel), mt_ch.ampcom);
    setParameter("ampcom_sc_" + std::to_string(channel), mt_ch.ampcom_sc);
    setParameter("cml_" + std::to_string(channel), mt_ch.cml);
    setParameter("cml_sc_" + std::to_string(channel), mt_ch.cml_sc);
    setParameter("amonctrl_" + std::to_string(channel), mt_ch.amonctrl);
    setParameter("comp_spi_" + std::to_string(channel), mt_ch.comp_spi);
    setParameter("tdctest_n_" + std::to_string(channel), mt_ch.tdctest_n);
    setParameter("sswitch_" + std::to_string(channel), mt_ch.sswitch);
    setParameter("delay_" + std::to_string(channel), mt_ch.delay);
    setParameter("pole_en_n_" + std::to_string(channel), mt_ch.pole_en_n);
    setParameter("energy_c_en_" + std::to_string(channel), mt_ch.energy_c_en);
    setParameter("energy_r_en_" + std::to_string(channel), mt_ch.energy_r_en);
    setParameter("cm_sensing_high_r_" + std::to_string(channel), mt_ch.cm_sensing_high_r);
    setParameter("amon_en_n_" + std::to_string(channel), mt_ch.amon_en_n);
    setParameter("edge_" + std::to_string(channel), mt_ch.edge);
    setParameter("edge_cml_" + std::to_string(channel), mt_ch.edge_cml);
    setParameter("dmon_en_" + std::to_string(channel), mt_ch.dmon_en);
    setParameter("dmon_sw_" + std::to_string(channel), mt_ch.dmon_sw);
}


} // namespace mutrig