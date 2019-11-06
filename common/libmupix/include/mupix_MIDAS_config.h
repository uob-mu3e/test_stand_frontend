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
"",\
NULL }
#endif


#ifndef MUPIX_CHIPDACS_DEFINED
#define MUPIX_CHIPDACS_DEFINED

typedef struct {
    INT       Bandgap1_on;
    INT       Biasblock1_on;
    INT       VNRegCasc;
    INT       VDel;
    INT       VPComp;
    INT       VPDAC;
    INT       BLResDig;
    INT       VPVCO;
    INT       VNVCO;
    INT       VPDelDclMux;
    INT       VNDelDclMux;
    INT       VPDelDcl;
    INT       VNDelDcl;
    INT       VPDelPreEmp;
    INT       VNDelPreEmp;
    INT       VPDcl;
    INT       VNDcl;
    INT       VNLVDS;
    INT       VNLVDSDel;
    INT       VPPump;
    INT       resetckdivend;
    INT       maxcycend;
    INT       slowdownend;
    INT       timerend;
    INT       tsphase;
    INT       ckdivend2;
    INT       ckdivend;
    INT       VPRegCasc;
    INT       VPRamp;
    INT       VPBiasReg;
    INT       VNBiasReg;
    INT       enable2threshold;
    INT       enableADC;
    INT       Invert;
    INT       SelEx;
    INT       SelSlow;
    INT       EnablePLL;
    INT       Readout_reset_n;
    INT       Serializer_reset_n;
    INT       Aurora_reset_n;
    INT       sendcounter;
    INT       Linkselect;
    INT       Termination;
    INT       AlwaysEnable;
    INT       SelectTest;
    INT       SelectTestOut;
    INT       DisableHitbus;
    INT       Bandgap2_on;
    INT       Biasblock2_on;
    INT       BLResPix;
    INT       VNPix;
    INT       VNFBPix;
    INT       VNFollPix;
    INT       VNPix2;
    INT       VNBiasPix;
    INT       VPLoadPix;
    INT       VNOutPix;
    INT       VPFoll;
    INT       VNDACPix;
    INT       ThLow;
    INT       ThPix;
    INT       BLPix;
    INT       BLDig;
    INT       ThHigh;
} MUPIX_CHIPDACS;

#define MUPIX_CHIPDACS_STR(_name) const char *_name[] = {\
"[.]",\
"Bandgap1_on = INT : 0",\
"Biasblock1_on = INT : 5",\
"VNRegCasc = INT : 0",\
"VDel = INT : 16",\
"VPComp = INT : 5",\
"VPDAC = INT : 0",\
"BLResDig = INT : 5",\
"VPVCO = INT : 12",\
"VNVCO = INT : 13",\
"VPDelDclMux = INT : 24",\
"VNDelDclMux = INT : 24",\
"VPDelDcl = INT : 40",\
"VNDelDcl = INT : 40",\
"VPDelPreEmp = INT : 24",\
"VNDelPreEmp = INT : 24",\
"VPDcl = INT : 24",\
"VNDcl = INT : 16",\
"VNLVDS = INT : 24",\
"VNLVDSDel = INT : 0",\
"VPPump = INT : 63",\
"resetckdivend = INT : 15",\
"maxcycend = INT : 63",\
"slowdownend = INT : 0",\
"timerend = INT : 1",\
"tsphase = INT : 0",\
"ckdivend2 = INT : 7",\
"ckdivend = INT : 0",\
"VPRegCasc = INT : 0",\
"VPRamp = INT : 0",\
"VPBiasReg = INT : 0",\
"VNBiasReg = INT : 0",\
"enable2threshold = INT : 0",\
"enableADC = INT : 1",\
"Invert = INT : 0",\
"SelEx = INT : 0",\
"SelSlow = INT : 0",\
"EnablePLL = INT : 1",\
"Readout_reset_n = INT : 1",\
"Serializer_reset_n = INT : 1",\
"Aurora_reset_n = INT : 1",\
"sendcounter = INT : 0",\
"Linkselect = INT : 1",\
"Termination = INT : 0",\
"AlwaysEnable = INT : 0",\
"SelectTest = INT : 0",\
"SelectTestOut = INT : 0",\
"DisableHitbus = INT : 1",\
"Bandgap2_on = INT : 0",\
"Biasblock2_on = INT : 5",\
"BLResPix = INT : 5",\
"VNPix = INT : 20",\
"VNFBPix = INT : 10",\
"VNFollPix = INT : 10",\
"VNPix2 = INT : 0",\
"VNBiasPix = INT : 0",\
"VPLoadPix = INT : 5",\
"VNOutPix = INT : 10",\
"VPFoll = INT : 10",\
"VNDACPix = INT : 0",\
"ThLow = INT : 312",\
"ThPix = INT : 454",\
"BLPix = INT : 448",\
"BLDig = INT : 256",\
"ThHigh = INT : 318",\
"",\
NULL }

typedef struct {
    INT RAM[128];
    BOOL EnableHitbus[128];
    BOOL EnableInjection[128];
} MUPIX_PIXELROWCONFIG;

#define MUPIX_PIXELROWCONFIG_STR(_name) const char *_name[] = {\
"[.]",\
"",\
NULL }
#endif

#endif //MU3EDAQ_MUPIX_MIDAS_CONFIG_H
