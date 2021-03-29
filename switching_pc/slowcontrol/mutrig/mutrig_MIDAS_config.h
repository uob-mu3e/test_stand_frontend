/********************************************************************\

  Name:         mutrig_MIDAS_config.h
  Created by:   Lukas Gerritzen

  Contents:     Contains the following structs from experim.h:
                MUTRIG_TDC
                MUTRIG_GLOBAL
                MUTRIG_CH

  Created on:   Mon Apr 29 2019

\********************************************************************/
#include <odbxx.h>
using midas::odb;

// From midas.h
// TODO: Why not include midas.h
typedef unsigned int DWORD;
typedef DWORD BOOL;
typedef int INT;


#ifndef EXCL_MUTRIG


#ifndef MU3EDAQ_MUTRIG_MIDAS_CONFIG_H
#define MU3EDAQ_MUTRIG_MIDAS_CONFIG_H

#ifndef MUTRIG_DAQ_DEFINED
#define MUTRIG_DAQ_DEFINED

typedef struct {
    BOOL dummy_config;
    BOOL dummy_data;
    INT  dummy_data_n;
    BOOL dummy_data_fast;
    BOOL prbs_decode_disable;
    BOOL reset_datapath;
    BOOL reset_asics;
} MUTRIG_DAQ;

static odb MUTRIG_DAQ_SETTINGS = {
    {"dummy_config", false},
    {"dummy_data", false},
    {"dummy_data_n", 200},
    {"dummy_data_fast", false},
    {"prbs_decode_disable", false},
    {"reset_datapath", false},
    {"reset_asics", false},
    {"reset_lvds", false},
    {"reset_counters", false},
    {"LVDS_waitforall", false},
    {"LVDS_waitforall_sticky", false},
    {"num_asics", 16},
    {"mask", { false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false
                }},
    {"resetskew_cphase", {false, false, false, false}},
    {"resetskew_cdelay", {false, false, false, false}},
    {"resetskew_phases", {false, false, false, false}},
};

#endif

#ifndef MUTRIG_TDC_DEFINED
#define MUTRIG_TDC_DEFINED

typedef struct {
    INT       vnpfc;
    INT       vnpfc_offset;
    BOOL      vnpfc_scale;
    INT       vncnt;
    INT       vncnt_offset;
    BOOL      vncnt_scale;
    INT       vnvcobuffer;
    INT       vnvcobuffer_offset;
    BOOL      vnvcobuffer_scale;
    INT       vnd2c;
    INT       vnd2c_offset;
    BOOL      vnd2c_scale;
    INT       vnpcp;
    INT       vnpcp_offset;
    BOOL      vnpcp_scale;
    INT       vnhitlogic;
    INT       vnhitlogic_offset;
    BOOL      vnhitlogic_scale;
    INT       vncntbuffer;
    INT       vncntbuffer_offset;
    BOOL      vncntbuffer_scale;
    INT       vnvcodelay;
    INT       vnvcodelay_offset;
    BOOL      vnvcodelay_scale;
    INT       latchbias;
    INT       ms_limits;
    BOOL      ms_switch_sel;
    BOOL      amon_en;
    INT       amon_dac;
    BOOL      dmon_1_en;
    INT       dmon_1_dac;
    BOOL      dmon_2_en;
    INT       dmon_2_dac;
    INT       lvds_tx_vcm;
    INT       lvds_tx_bias;
    BOOL      coin_xbar_lower_rx_ena;
    BOOL      coin_xbar_lower_tx_ena;
    INT       coin_xbar_lower_tx_vdac;
    INT       coin_xbar_lower_tx_idac;
    INT       coin_mat_xbl;
    INT       coin_mat_xbu;
    BOOL      coin_xbar_upper_rx_ena;
    BOOL      coin_xbar_upper_tx_ena;
    INT       coin_xbar_upper_tx_vdac;
    INT       coin_xbar_upper_tx_idac;
} MUTRIG_TDC;

static odb MUTRIG_TDC_SETTINGS = {
    {"vnpfc", 63},
    {"vnpfc_offset", 3},
    {"vnpfc_scale", false},
    {"vncnt", 63},
    {"vncnt_offset", 3},
    {"vncnt_scale", false},
    {"vnvcobuffer", 63},
    {"vnvcobuffer_offset", 3},
    {"vnvcobuffer_scale", false},
    {"vnd2c", 3},
    {"vnd2c_offset", 3},
    {"vnd2c_scale", false},
    {"vnpcp", 63},
    {"vnpcp_offset", 3},
    {"vnpcp_scale", false},
    {"vnhitlogic", 63},
    {"vnhitlogic_offset", 3},
    {"vnhitlogic_scale", false},
    {"vncntbuffer", 63}, 
    {"vncntbuffer_offset", 3},
    {"vncntbuffer_scale", false},
    {"vnvcodelay", 63},
    {"vnvcodelay_offset", 3},
    {"vnvcodelay_scale", false},
    {"latchbias", 0},
    {"ms_limits", 0},
    {"ms_switch_sel", false},
    {"amon_en", false},
    {"amon_dac", 0},
    {"dmon_1_en", false},
    {"dmon_1_dac", 0},
    {"dmon_2_en", false},
    {"dmon_2_dac", 0},
    {"lvds_tx_vcm", 0},
    {"lvds_tx_bias", 0},
    {"coin_xbar_lower_rx_ena", false},
    {"coin_xbar_lower_tx_ena", false},
    {"coin_xbar_lower_tx_vdac", 0},
    {"coin_xbar_lower_tx_idac", 0},
    {"coin_mat_xbl", 0},
    {"coin_mat_xbu", 0},
    {"coin_xbar_upper_rx_ena", false},
    {"coin_xbar_upper_tx_ena", false},
    {"coin_xbar_upper_tx_vdac", 0},
    {"coin_xbar_upper_tx_idac", 0},
};

#endif


#ifndef MUTRIG_GLOBAL_DEFINED
#define MUTRIG_GLOBAL_DEFINED

typedef struct {
    BOOL      ext_trig_mode;
    BOOL      ext_trig_endtime_sign;
    INT       ext_trig_offset;
    INT       ext_trig_endtime;
    BOOL      gen_idle;
    BOOL      ms_debug;
    BOOL      prbs_debug;
    BOOL      prbs_single;
    BOOL      sync_ch_rst;
    BOOL      disable_coarse;
    BOOL      pll_setcoarse;
    BOOL      short_event_mode;
    BOOL      pll_envomonitor;
} MUTRIG_GLOBAL;

static odb MUTRIG_GLOBAL_SETTINGS = {
    {"ext_trig_mode", false},
    {"ext_trig_endtime_sign", false},
    {"ext_trig_offset", 32},
    {"ext_trig_endtime", 32},
    {"gen_idle", false},
    {"ms_debug", false},
    {"prbs_debug", false},
    {"rbs_single", false},
    {"sync_ch_rst", false},
    {"disable_coarse", false},
    {"pll_setcoarse", false},
    {"short_event_mode", false},
    {"pll_envomonitor", false},
};

#endif

#ifndef MUTRIG_CH_DEFINED
#define MUTRIG_CH_DEFINED

typedef struct {
    BOOL mask;
    BOOL recv_all;
    INT tthresh;
    INT tthresh_sc;
    INT ethresh;
    INT sipm;
    INT sipm_sc;
    INT inputbias;
    INT inputbias_sc;
    INT pole;
    INT pole_sc;
    INT ampcom;
    INT ampcom_sc;
    INT cml;
    INT cml_sc;
    INT amonctrl;
    INT comp_spi;
    INT coin_mat;
    BOOL tdctest_n;
    BOOL sswitch;
    BOOL delay;
    BOOL pole_en_n;
    BOOL energy_c_en;
    BOOL energy_r_en;
    BOOL cm_sensing_high_r;
    BOOL amon_en_n;
    BOOL edge;
    BOOL edge_cml;
    BOOL dmon_en;
    BOOL dmon_sw;
} MUTRIG_CH;

static odb MUTRIG_CH_SETTINGS = {
    {"mask", false},
    {"recv_all", false},
    {"tthresh", 0},
    {"tthresh_sc", 0},
    {"ethresh", 0},
    {"sipm", 0},
    {"sipm_sc", 0},
    {"inputbias", 0},
    {"inputbias_sc", 0},
    {"pole", 0},
    {"pole_sc", 0},
    {"ampcom", 0},
    {"ampcom_sc", 0},
    {"cml", 0},
    {"cml_sc", 0},
    {"amonctrl", 0},
    {"comp_spi", 0},
    {"coin_mat", 0},
    {"tdctest_n", false},
    {"sswitch", false},
    {"delay", false},
    {"pole_en_n", false},
    {"energy_c_en", false},
    {"energy_r_en", false},
    {"cm_sensing_high_r", false},
    {"amon_en_n", false},
    {"edge", false},
    {"edge_cml", false},
    {"dmon_en", false},
    {"dmon_sw", false},
};
#endif


#endif //MU3EDAQ_MUTRIG_MIDAS_CONFIG_H
#endif //EXCL_MUTRIG
