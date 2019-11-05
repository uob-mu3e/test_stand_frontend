/********************************************************************\

  Name:         mupix_MIDAS_config.h
  Created by:   Lukas Gerritzen

  Contents:     Contains the following structs from experim.h:
                MUPIX_TDC
                MUPIX_GLOBAL
                MUPIX_CH

  Created on:   Mon Apr 29 2019

\********************************************************************/
// From midas.h
typedef unsigned int DWORD;
typedef DWORD BOOL;
typedef int INT;


#ifndef MU3EDAQ_MUPIX_MIDAS_CONFIG_H
#define MU3EDAQ_MUPIX_MIDAS_CONFIG_H

#ifndef MUTRIG_GLOBAL_DEFINED
#define MUTRIG_GLOBAL_DEFINED

typedef struct {
    INT       n_asics;
} MUTRIG_GLOBAL;

#define MUTRIG_GLOBAL_STR(_name) const char *_name[] = {\
"[.]",\
"Num asics = INT : 1",\
"Num boards = INT : 1",\
"",\
NULL }

#endif


#ifndef MUPIX_DAQ_DEFINED
#define MUPIX_DAQ_DEFINED

typedef struct {
  BOOL dummy_config;
  BOOL dummy_data;
} MUPIX_DAQ;

#define MUPIX_DAQ_STR(_name) const char *_name[] = {\
"[.]",\
"dummy_config = BOOL : n",\
"dummy_data = BOOL : n",\
"dummy_data_n = INT : 255",\
"dummy_data_fast = BOOL : n",\
"prbs_decode_bypass = BOOL : n",\
"reset_datapath = BOOL : n",\
"reset_asics = BOOL : n",\
"mask = BOOL[16] n",\
"",\
NULL }

#endif

#ifndef MUPIX_BOARDDACS_DEFINED
#define MUPIX_BOARDDACS_DEFINED

typedef struct {
    BOOL      amon_en;
    INT       amon_dac;
    BOOL      dmon_1_en;
    INT       dmon_1_dac;
    BOOL      dmon_2_en;
    INT       dmon_2_dac;
} MUPIX_BOARDDACS;

#define MUPIX_BOARDDACS_STR(_name) const char *_name[] = {\
"[.]",\
"amon_en = BOOL : n",\
"amon_dac = INT : 0",\
"dmon_1_en = BOOL : n",\
"dmon_1_dac = INT : 0",\
"dmon_2_en = BOOL : n",\
"dmon_2_dac = INT : 0",\
"",\
NULL }
#endif


#ifndef MUPIX_CHIPDACS_DEFINED
#define MUPIX_CHIPDACS_DEFINED

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
} MUPIX_CHIPDACS;

#define MUPIX_CHIPDACS_STR(_name) const char *_name[] = {\
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
"ms_limits = INT : 0",\
"ms_switch_sel = BOOL : n",\
"amon_en = BOOL : n",\
"amon_dac = INT : 0",\
"dmon_1_en = BOOL : n",\
"dmon_1_dac = INT : 0",\
"dmon_2_en = BOOL : n",\
"dmon_2_dac = INT : 0",\
"",\
NULL }
#endif

#endif //MU3EDAQ_MUPIX_MIDAS_CONFIG_H
