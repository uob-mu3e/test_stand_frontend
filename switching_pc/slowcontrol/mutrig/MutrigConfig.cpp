#include <cstring>
#include <iostream>
#include <iomanip>

#include "MutrigConfig.h"
#include "odbxx.h"

using midas::odb;

namespace mutrig {

MutrigConfig::paras_t MutrigConfig::parameters_tdc = {
        make_param("vnd2c_scale",        1, 1),
        make_param("vnd2c_offset",       2, 1),
        make_param("vnd2c",              6, 1),
        make_param("vncntbuffer_scale",  1, 1),
        make_param("vncntbuffer_offset", 2, 1),
        make_param("vncntbuffer",        6, 1),
        make_param("vncnt_scale",        1, 1),
        make_param("vncnt_offset",       2, 1),
        make_param("vncnt",              6, 1),
        make_param("vnpcp_scale",        1, 1),
        make_param("vnpcp_offset",       2, 1),
        make_param("vnpcp",              6, 1),
        make_param("vnvcodelay_scale",   1, 1),
        make_param("vnvcodelay_offset",  2, 1),
        make_param("vnvcodelay",         6, 1),
        make_param("vnvcobuffer_scale",  1, 1),
        make_param("vnvcobuffer_offset", 2, 1),
        make_param("vnvcobuffer",        6, 1),
        make_param("vnhitlogic_scale",   1, 1),
        make_param("vnhitlogic_offset",  2, 1),
        make_param("vnhitlogic",         6, 1),
        make_param("vnpfc_scale",        1, 1),
        make_param("vnpfc_offset",       2, 1),
        make_param("vnpfc",              6, 1),
        make_param("latchbias",          12, 0)
    };

MutrigConfig::paras_t MutrigConfig::parameters_ch = {
        make_param("energy_c_en",       1, 1), //old name: anode_flag
        make_param("energy_r_en",       1, 1), //old name: cathode_flag
        make_param("sswitch",           1, 1),
        make_param("cm_sensing_high_r", 1, 1), //old name: SorD; should be always '0'
        make_param("amon_en_n",         1, 1), //old name: SorD_not; 0: enable amon in the channel
        make_param("edge",              1, 1),
        make_param("edge_cml",          1, 1),
        make_param("cml_sc",            1, 1),
        make_param("dmon_en",           1, 1),
        make_param("dmon_sw",           1, 1),
        make_param("tdctest_n",         1, 1),
        make_param("amonctrl",          3, 1),
        make_param("comp_spi",          2, 1),
        make_param("sipm_sc",           1, 1),
        make_param("sipm",              6, 1),
        make_param("tthresh_sc",        3, 1),
        make_param("tthresh",           6, 1),
        make_param("ampcom_sc",         2, 1),
        make_param("ampcom",            6, 1),
        make_param("inputbias_sc",      1, 1),
        make_param("inputbias",         6, 1),
        make_param("ethresh",           8, 1),
        make_param("ebias",             3, 1),
        make_param("pole_sc",           1, 1),
        make_param("pole",              6, 1),
        make_param("cml",               4, 1),
        make_param("delay",             1, 1),
        make_param("pole_en_n",         1, 1), //old name: dac_delay_bit1; 0: DAC_pole on
        make_param("mask",              1, 1),
        make_param("recv_all",          1, 1) //new in mutrig2
     };

MutrigConfig::paras_t MutrigConfig::parameters_header = {
        make_param("gen_idle",              1, 1),
        make_param("sync_ch_rst",           1, 1), // mutrig2: was recv_all, now setting to enable on-chip reset synchronizer
        make_param("ext_trig_mode",         1, 1), // new
        make_param("ext_trig_endtime_sign", 1, 1), // sign of the external trigger matching window, 1: end time is after the trigger; 0: end time is before the trigger
        make_param("ext_trig_offset",       4, 0), // offset of the external trigger matching window
        make_param("ext_trig_endtime",      4, 0), // end time of external trigger matching window
        make_param("ms_limits",             5, 0),
        make_param("ms_switch_sel",         1, 1),
        make_param("ms_debug",              1, 1),
        make_param("prbs_debug",            1, 1), // new
        make_param("prbs_single",           1, 1), // new
        make_param("short_event_mode",      1, 1), //fast transmission mode
        make_param("pll_setcoarse",         1, 1),
        make_param("pll_envomonitor",       1, 1),
        make_param("disable_coarse",        1, 1)
    };

MutrigConfig::paras_t MutrigConfig::parameters_footer = {
//coincidence logic crossbar / lower
        make_param("coin_xbar_lower_rx_ena",  1, 1),
        make_param("coin_xbar_lower_tx_ena",  1, 1),
        make_param("coin_xbar_lower_tx_vdac", 8, 1),
        make_param("coin_xbar_lower_tx_idac", 6, 1),
//coincidence logic matrix
        make_param("coin_mat_xbl", 3, 1),
        make_param("coin_mat_0", 3, 1),
        make_param("coin_mat_1", 3, 1),
        make_param("coin_mat_2", 3, 1),
        make_param("coin_mat_3", 3, 1),
        make_param("coin_mat_4", 3, 1),
        make_param("coin_mat_5", 3, 1),
        make_param("coin_mat_6", 3, 1),
        make_param("coin_mat_7", 3, 1),
        make_param("coin_mat_8", 3, 1),
        make_param("coin_mat_9", 3, 1),
        make_param("coin_mat_10", 3, 1),
        make_param("coin_mat_11", 3, 1),
        make_param("coin_mat_12", 3, 1),
        make_param("coin_mat_13", 3, 1),
        make_param("coin_mat_14", 3, 1),
        make_param("coin_mat_15", 3, 1),
        make_param("coin_mat_16", 3, 1),
        make_param("coin_mat_17", 3, 1),
        make_param("coin_mat_18", 3, 1),
        make_param("coin_mat_19", 3, 1),
        make_param("coin_mat_20", 3, 1),
        make_param("coin_mat_21", 3, 1),
        make_param("coin_mat_22", 3, 1),
        make_param("coin_mat_23", 3, 1),
        make_param("coin_mat_24", 3, 1),
        make_param("coin_mat_25", 3, 1),
        make_param("coin_mat_26", 3, 1),
        make_param("coin_mat_27", 3, 1),
        make_param("coin_mat_28", 3, 1),
        make_param("coin_mat_29", 3, 1),
        make_param("coin_mat_30", 3, 1),
        make_param("coin_mat_31", 3, 1),
        make_param("coin_mat_xbu", 3, 1),
//coincidence logic crossbar / upper
        make_param("coin_xbar_upper_rx_ena",  1, 1),
        make_param("coin_xbar_upper_tx_ena",  1, 1),
        make_param("coin_xbar_upper_tx_vdac", 8, 1),
        make_param("coin_xbar_upper_tx_idac", 6, 1),

        make_param("amon_en",       1, 1),
        make_param("amon_dac",      8, 1),
        make_param("dmon_1_en",     1, 1),
        make_param("dmon_1_dac",    8, 1),
        make_param("dmon_2_en",     1, 1),
        make_param("dmon_2_dac",    8, 1),
        make_param("lvds_tx_vcm",   8, 1), // new
        make_param("lvds_tx_bias",  6, 1)  // new
    };


MutrigConfig::MutrigConfig() {
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

MutrigConfig::~MutrigConfig() {
    delete[] bitpattern_r;
    delete[] bitpattern_w;
}

void MutrigConfig::setParameterODBpp(std::string paraName, odb o){
    setParameter(paraName, o[paraName]);
}

void MutrigConfig::setParameterODBpp(std::string paraName, odb o, int channel){
    setParameter(paraName + "_" + std::to_string(channel) , o[paraName]);
}

void MutrigConfig::Parse_GLOBAL_from_struct(odb o){
    //hard coded in order to avoid macro magic
//    setParameter("", mt_g.n_asics);
//    setParameter("", mt_g.n_channels);
    MutrigConfig::setParameterODBpp("ext_trig_mode", o);
    MutrigConfig::setParameterODBpp("ext_trig_endtime_sign", o);
    MutrigConfig::setParameterODBpp("ext_trig_offset", o);
    MutrigConfig::setParameterODBpp("ext_trig_endtime", o);
    MutrigConfig::setParameterODBpp("gen_idle", o);
    MutrigConfig::setParameterODBpp("ms_debug", o);
    MutrigConfig::setParameterODBpp("prbs_debug", o);
    MutrigConfig::setParameterODBpp("prbs_single", o);
    MutrigConfig::setParameterODBpp("sync_ch_rst", o);
    MutrigConfig::setParameterODBpp("disable_coarse", o);
    MutrigConfig::setParameterODBpp("pll_setcoarse", o);
    MutrigConfig::setParameterODBpp("short_event_mode", o);
    MutrigConfig::setParameterODBpp("pll_envomonitor", o);
}

void MutrigConfig::Parse_TDC_from_struct(odb o){
    MutrigConfig::setParameterODBpp("vnpfc", o);
    MutrigConfig::setParameterODBpp("vnpfc_offset", o);
    MutrigConfig::setParameterODBpp("vnpfc_scale", o);
    MutrigConfig::setParameterODBpp("vncnt", o);
    MutrigConfig::setParameterODBpp("vncnt_offset", o);
    MutrigConfig::setParameterODBpp("vncnt_scale", o);
    MutrigConfig::setParameterODBpp("vnvcobuffer", o);
    MutrigConfig::setParameterODBpp("vnvcobuffer_offset", o);
    MutrigConfig::setParameterODBpp("vnvcobuffer_scale", o);
    MutrigConfig::setParameterODBpp("vnd2c", o);
    MutrigConfig::setParameterODBpp("vnd2c_offset", o);
    MutrigConfig::setParameterODBpp("vnd2c_scale", o);
    MutrigConfig::setParameterODBpp("vnpcp", o);
    MutrigConfig::setParameterODBpp("vnpcp_offset", o);
    MutrigConfig::setParameterODBpp("vnpcp_scale", o);
    MutrigConfig::setParameterODBpp("vnhitlogic", o);
    MutrigConfig::setParameterODBpp("vnhitlogic_offset", o);
    MutrigConfig::setParameterODBpp("vnhitlogic_scale", o);
    MutrigConfig::setParameterODBpp("vncntbuffer", o);
    MutrigConfig::setParameterODBpp("vncntbuffer_offset", o);
    MutrigConfig::setParameterODBpp("vncntbuffer_scale", o);
    MutrigConfig::setParameterODBpp("vnvcodelay", o);
    MutrigConfig::setParameterODBpp("vnvcodelay_offset", o);
    MutrigConfig::setParameterODBpp("vnvcodelay_scale", o);
    MutrigConfig::setParameterODBpp("latchbias", o);
    MutrigConfig::setParameterODBpp("ms_limits", o);
    MutrigConfig::setParameterODBpp("ms_switch_sel", o);
    MutrigConfig::setParameterODBpp("amon_en", o);
    MutrigConfig::setParameterODBpp("amon_dac", o);
    MutrigConfig::setParameterODBpp("dmon_1_en", o);
    MutrigConfig::setParameterODBpp("dmon_1_dac", o);
    MutrigConfig::setParameterODBpp("dmon_2_en", o);
    MutrigConfig::setParameterODBpp("dmon_2_dac", o);
    MutrigConfig::setParameterODBpp("lvds_tx_vcm", o);
    MutrigConfig::setParameterODBpp("lvds_tx_bias", o);
    MutrigConfig::setParameterODBpp("coin_xbar_lower_rx_ena", o);
    MutrigConfig::setParameterODBpp("coin_xbar_lower_tx_ena", o);
    MutrigConfig::setParameterODBpp("coin_xbar_lower_tx_vdac", o);
    MutrigConfig::setParameterODBpp("coin_xbar_lower_tx_idac", o);
    MutrigConfig::setParameterODBpp("coin_xbar_upper_rx_ena", o);
    MutrigConfig::setParameterODBpp("coin_xbar_upper_tx_ena", o);
    MutrigConfig::setParameterODBpp("coin_xbar_upper_tx_vdac", o);
    MutrigConfig::setParameterODBpp("coin_xbar_upper_tx_idac", o);
    MutrigConfig::setParameterODBpp("coin_mat_xbl", o);
    MutrigConfig::setParameterODBpp("coin_mat_xbu", o);
}


void MutrigConfig::Parse_CH_from_struct(odb o, int channel){
<<<<<<< HEAD
    MutrigConfig::setParameterODBpp("mask", o, channel);
    MutrigConfig::setParameterODBpp("recv_all", o, channel);
    MutrigConfig::setParameterODBpp("tthresh", o, channel);
    MutrigConfig::setParameterODBpp("tthresh_sc", o, channel);
    MutrigConfig::setParameterODBpp("ethresh", o, channel);
    MutrigConfig::setParameterODBpp("ebias", o, channel);
    MutrigConfig::setParameterODBpp("sipm", o, channel);
    MutrigConfig::setParameterODBpp("sipm_sc", o, channel);
    MutrigConfig::setParameterODBpp("inputbias", o, channel);
    MutrigConfig::setParameterODBpp("inputbias_sc", o, channel);
    MutrigConfig::setParameterODBpp("pole", o, channel);
    MutrigConfig::setParameterODBpp("pole_sc", o, channel);
    MutrigConfig::setParameterODBpp("ampcom", o, channel);
    MutrigConfig::setParameterODBpp("ampcom_sc", o, channel);
    MutrigConfig::setParameterODBpp("cml", o, channel);
    MutrigConfig::setParameterODBpp("cml_sc", o, channel);
    MutrigConfig::setParameterODBpp("amonctrl", o, channel);
    MutrigConfig::setParameterODBpp("comp_spi", o, channel);
    MutrigConfig::setParameterODBpp("tdctest_n", o, channel);
    MutrigConfig::setParameterODBpp("sswitch", o, channel);
    MutrigConfig::setParameterODBpp("delay", o, channel);
    MutrigConfig::setParameterODBpp("pole_en_n", o, channel);
    MutrigConfig::setParameterODBpp("energy_c_en", o, channel);
    MutrigConfig::setParameterODBpp("energy_r_en", o, channel);
    MutrigConfig::setParameterODBpp("cm_sensing_high_r", o, channel);
    MutrigConfig::setParameterODBpp("amon_en_n", o, channel);
    MutrigConfig::setParameterODBpp("edge", o, channel);
    MutrigConfig::setParameterODBpp("edge_cml", o, channel);
    MutrigConfig::setParameterODBpp("dmon_en", o, channel);
    MutrigConfig::setParameterODBpp("dmon_sw", o, channel);
    MutrigConfig::setParameterODBpp("coin_mat", o, channel);
=======
    MutrigConfig::setParameterODBpp("mask_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("recv_all_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("tthresh_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("tthresh_sc_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("ethresh_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("ebias_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("sipm_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("sipm_sc_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("inputbias_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("inputbias_sc_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("pole_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("pole_sc_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("ampcom_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("ampcom_sc_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("cml_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("cml_sc_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("amonctrl_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("comp_spi_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("tdctest_n_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("sswitch_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("delay_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("pole_en_n_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("energy_c_en_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("energy_r_en_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("cm_sensing_high_r_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("amon_en_n_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("edge_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("edge_cml_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("dmon_en_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("dmon_sw_" + std::to_string(channel), o);
    MutrigConfig::setParameterODBpp("coin_mat_" + std::to_string(channel), o);
>>>>>>> origin/Scifi_FEB_v2
}


} // namespace mutrig
