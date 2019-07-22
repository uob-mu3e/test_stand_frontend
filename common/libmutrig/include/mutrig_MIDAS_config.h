/********************************************************************\

  Name:         mutrig_MIDAS_config.h
  Created by:   Lukas Gerritzen

  Contents:     Contains the following structs from experim.h:
                MUTRIG_TDC
                MUTRIG_GLOBAL
                MUTRIG_CH

  Created on:   Mon Apr 29 2019

\********************************************************************/
// From midas.h
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
  BOOL prbs_decode_enable;
} MUTRIG_DAQ;

#define MUTRIG_DAQ_STR(_name) const char *_name[] = {\
"[.]",\
"dummy_config = BOOL : n",\
"dummy_data = BOOL : n",\
"dummy_data_n = INT : 255",\
"dummy_data_fast = BOOL : n",\
"prbs_decode_enable = BOOL : n",\
"mask = BOOL : n",\
"",\
NULL }

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
    BOOL      amon_en;
    INT       amon_dac;
    BOOL      dmon_1_en;
    INT       dmon_1_dac;
    BOOL      dmon_2_en;
    INT       dmon_2_dac;
} MUTRIG_TDC;

#define MUTRIG_TDC_STR(_name) const char *_name[] = {\
"[.]",\
"vnpfc = INT : 63",\
"vnpfc_offset = INT : 3",\
"vnpfc_scale = BOOL : n",\
"vncnt = INT : 63",\
"vncnt_offset = INT : 3",\
"vncnt_scale = BOOL : n",\
"vnvcobuffer = INT : 63",\
"vnvcobuffer_offset = INT : 3",\
"vnvcobuffer_scale = BOOL : n",\
"vnd2c = INT : 63",\
"vnd2c_offset = INT : 3",\
"vnd2c_scale = BOOL : n",\
"vnpcp = INT : 63",\
"vnpcp_offset = INT : 3",\
"vnpcp_scale = BOOL : n",\
"vnhitlogic = INT : 63",\
"vnhitlogic_offset = INT : 3",\
"vnhitlogic_scale = BOOL : n",\
"vncntbuffer = INT : 63",\
"vncntbuffer_offset = INT : 3",\
"vncntbuffer_scale = BOOL : n",\
"vnvcodelay = INT : 63",\
"vnvcodelay_offset = INT : 3",\
"vnvcodelay_scale = BOOL : n",\
"latchbias = INT : 0",\
"amon_en = BOOL : n",\
"amon_dac = INT : 0",\
"dmon_1_en = BOOL : n",\
"dmon_1_dac = INT : 0",\
"dmon_2_en = BOOL : n",\
"dmon_2_dac = INT : 0",\
"",\
NULL }

#endif


#ifndef MUTRIG_GLOBAL_DEFINED
#define MUTRIG_GLOBAL_DEFINED

typedef struct {
    INT       n_asics;
    BOOL      ext_trig_mode;
    BOOL      ext_trig_endtime_sign;
    INT       ext_trig_offset;
    INT       ext_trig_endtime;
    BOOL      gen_idle;
    BOOL      ms_switch_sel;
    BOOL      ms_debug;
    BOOL      prbs_debug;
    BOOL      prbs_single;
    BOOL      recv_all;
    BOOL      disable_coarse;
    BOOL      pll_setcoarse;
    BOOL      short_event_mode;
    INT       ms_limits;
    BOOL      pll_envomonitor;
    INT       lvds_tx_vcm;
    INT       lvds_tx_bias;
} MUTRIG_GLOBAL;

#define MUTRIG_GLOBAL_STR(_name) const char *_name[] = {\
"[.]",\
"Num asics = INT : 1",\
"ext_trig_mode = BOOL : n",\
"ext_trig_endtime_sign = BOOL : n",\
"ext_trig_offset = INT : 32",\
"ext_trig_endtime = INT : 32",\
"gen_idle = BOOL : n",\
"ms_switch_sel = BOOL : n",\
"ms_debug = BOOL : n",\
"prbs_debug = BOOL : n",\
"prbs_single = BOOL : n",\
"recv_all = BOOL : n",\
"disable_coarse = BOOL : n",\
"pll_setcoarse = BOOL : n",\
"short_event_mode = BOOL : n",\
"ms_limits = INT : 0",\
"pll_envomonitor = BOOL : n",\
"lvds_tx_vcm = INT : 0",\
"lvds_tx_bias = INT : 0",\
"",\
NULL }

#endif

#ifndef MUTRIG_CH_DEFINED
#define MUTRIG_CH_DEFINED

typedef struct {
    BOOL mask;
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

#define MUTRIG_CH_STR(_name) const char *_name[] = {\
"[.]",\
"mask = BOOL : 0",\
"tthresh = INT : 0",\
"tthresh_sc = INT : 0",\
"ethresh = INT : 0",\
"sipm = INT : 0",\
"sipm_sc = INT : 0",\
"inputbias = INT : 0",\
"inputbias_sc = INT : 0",\
"pole = INT : 0",\
"pole_sc = INT : 0",\
"ampcom = INT : 0",\
"ampcom_sc = INT : 0",\
"cml = INT : 0",\
"cml_sc = INT : 0",\
"amonctrl = INT : 0",\
"comp_spi = INT : 0",\
"tdctest_n = BOOL : 0",\
"sswitch = BOOL : 0",\
"delay = BOOL : 0",\
"pole_en_n = BOOL : 0",\
"energy_c_en = BOOL : 0",\
"energy_r_en = BOOL : 0",\
"cm_sensing_high_r = BOOL : 0",\
"amon_en_n = BOOL : 0",\
"edge = BOOL : 0",\
"edge_cml = BOOL : 0",\
"dmon_en = BOOL : 0",\
"dmon_sw = BOOL : 0",\
"",\
NULL }

#endif


#endif //MU3EDAQ_MUTRIG_MIDAS_CONFIG_H
#endif //EXCL_MUTRIG
