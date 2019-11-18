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
    INT       n_boards;
} MUTRIG_GLOBAL;

#define MUPIX_GLOBAL_STR(_name) const char *_name[] = {\
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
    int       Bandgap1_on;
    int       Biasblock1_on;
    int       unused_1;
    int       unused_2;
    int       unused_3;
    int       unused_4;
    int       unused_5;
    int       VNRegCasc;
    int       VDel;
    int       VPComp;
    int       VPDAC;
    int       unused_6;
    int       BLResDig;
    int       unused_7;
    int       unused_8;
    int       unused_9;
    int       VPVCO;
    int       VNVCO;
    int       VPDelDclMux;
    int       VNDelDclMux;
    int       VPDelDcl;
    int       VNDelDcl;
    int       VPDelPreEmp;
    int       VNDelPreEmp;
    int       VPDcl;
    int       VNDcl;
    int       VNLVDS;
    int       VNLVDSDel;
    int       VPPump;
    int       resetckdivend;
    int       maxcycend;
    int       slowdownend;
    int       timerend;
    int       tsphase;
    int       ckdivend2;
    int       ckdivend;
    int       VPRegCasc;
    int       VPRamp;
    int       unused_10;
    int       unused_11;
    int       unused_12;
    int       VPBiasReg;
    int       VNBiasReg;
    int       enable2threshold;
    int       enableADC;
    int       Invert;
    int       SelEx;
    int       SelSlow;
    int       EnablePLL;
    int       Readout_reset_n;
    int       Serializer_reset_n;
    int       Aurora_reset_n;
    int       sendcounter;
    int       Linkselect;
    int       Termination;
    int       AlwaysEnable;
    int       SelectTest;
    int       SelectTestOut;
    int       DisableHitbus;
    int       unused_13;
} MUPIX_CHIPDACS;

#define MUPIX_CHIPDACS_STR(_name) const char *_name[] = {\
"[.]",\
"Bandgap1_on = INT : 0",\
"Biasblock1_on = INT : 5",\
"unused_1 = INT : 0",\
"unused_2 = INT : 0",\
"unused_3 = INT : 0",\
"unused_4 = INT : 0",\
"unused_5 = INT : 0",\
"VNRegCasc = INT : 0",\
"VDel = INT : 16",\
"VPComp = INT : 5",\
"VPDAC = INT : 0",\
"unused_6 = INT : 0",\
"BLResDig = INT : 5",\
"unused_7 = INT : 0",\
"unused_8 = INT : 0",\
"unused_9 = INT : 0",\
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
"unused_10 = INT : 0",\
"unused_11 = INT : 0",\
"unused_12 = INT : 0",\
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
"unused_13 = INT : 0",\
"",\
NULL }

typedef  struct {
    int       digiWrite;
} MUPIX_DIGIROWDACS;

#define MUPIX_DIGIROWDACS_STR(_name) const char *_name[] = {\
"[.]",\
"digiWrite = INT : 0",\
"",\
NULL }

typedef  struct {
    int       RAM;
    int       EnableHitbus;
    int       EnableInjection;
} MUPIX_COLDACS;

#define MUPIX_COLDACS_STR(_name) const char *_name[] = {\
"[.]",\
"RAM = INT : 0",\
"EnableHitbus = INT : 0",\
"EnableInjection = INT : 0",\
"",\
NULL }

typedef  struct {
    int       unused;
    int       EnableInjection;
    int       EnableAnalogueBuffer;
} MUPIX_ROWDACS;

#define MUPIX_ROWDACS_STR(_name) const char *_name[] = {\
"[.]",\
"unused = INT : 0",\
"EnableInjection = INT : 0",\
"EnableAnalogueBuffer = INT : 0",\
"",\
NULL }

typedef struct {
    int       Bandgap2_on;
    int       Biasblock2_on;
    int       BLResPix;
    int       unused_1;
    int       VNPix;
    int       VNFBPix;
    int       VNFollPix;
    int       unused_2;
    int       unused_3;
    int       unused_4;
    int       unused_5;
    int       VNPix2;
    int       unused_6;
    int       VNBiasPix;
    int       VPLoadPix;
    int       VNOutPix;
    int       unused_7;
    int       unused_8;
    int       unused_9;
    int       unused_10;
    int       unused_11;
    int       unused_12;
    int       unused_13;
    int       unused_14;
    int       unused_15;
    int       unused_16;
    int       unused_17;
    int       unused_18;
    int       unused_19;
    int       unused_20;
    int       unused_21;
    int       unused_22;
    int       unused_23;
    int       unused_24;
    int       unused_25;
    int       unused_26;
    int       unused_27;
    int       unused_28;
    int       unused_29;
    int       VPFoll;
    int       VNDACPix;
    int       unused_30;
    int       unused_31;
    int       unused_32;
    int       unused_33;
    int       unused_34;
    int       unused_35;
    int       unused_36;
    int       unused_37;
    int       unused_38;
    int       unused_39;
    int       unused_40;
    int       unused_41;
    int       unused_42;
    int       unused_43;
    int       unused_44;
    int       unused_45;
    int       unused_46;
    int       unused_47;
    int       unused_48;
} MUPIX_CHIPDACS2;

#define MUPIX_CHIPDACS2_STR(_name) const char *_name[] = {\
"[.]",\
"Bandgap2_on = INT : 0",\
"Biasblock2_on = INT : 5",\
"BLResPix = INT : 5",\
"unused_1 = INT : 0",\
"VNPix = INT : 20",\
"VNFBPix = INT : 10",\
"VNFollPix = INT : 10",\
"unused_2 = INT : 0",\
"unused_3 = INT : 0",\
"unused_4 = INT : 0",\
"unused_5 = INT : 0",\
"VNPix2 = INT : 0",\
"unused_6 = INT : 0",\
"VNBiasPix = INT : 0",\
"VPLoadPix = INT : 5",\
"VNOutPix = INT : 10",\
"unused_7 = INT : 0",\
"unused_8 = INT : 0",\
"unused_9 = INT : 0",\
"unused_10 = INT : 0",\
"unused_11 = INT : 0",\
"unused_12 = INT : 0",\
"unused_13 = INT : 0",\
"unused_14 = INT : 0",\
"unused_15 = INT : 0",\
"unused_16 = INT : 0",\
"unused_17 = INT : 0",\
"unused_18 = INT : 0",\
"unused_19 = INT : 0",\
"unused_20 = INT : 0",\
"unused_21 = INT : 0",\
"unused_22 = INT : 0",\
"unused_23 = INT : 0",\
"unused_24 = INT : 0",\
"unused_25 = INT : 0",\
"unused_26 = INT : 0",\
"unused_27 = INT : 0",\
"unused_28 = INT : 0",\
"unused_29 = INT : 0",\
"VPFoll = INT : 10",\
"VNDACPix = INT : 0",\
"unused_30 = INT : 0",\
"unused_31 = INT : 0",\
"unused_32 = INT : 0",\
"unused_33 = INT : 0",\
"unused_34 = INT : 0",\
"unused_35 = INT : 0",\
"unused_36 = INT : 0",\
"unused_37 = INT : 0",\
"unused_38 = INT : 0",\
"unused_39 = INT : 0",\
"unused_40 = INT : 0",\
"unused_41 = INT : 0",\
"unused_42 = INT : 0",\
"unused_43 = INT : 0",\
"unused_44 = INT : 0",\
"unused_45 = INT : 0",\
"unused_46 = INT : 0",\
"unused_47 = INT : 0",\
"unused_48 = INT : 0",\
NULL }

typedef struct {
    int       ThLow;
    int       ThPix;
    int       BLPix;
    int       BLDig;
    int       ThHigh;
} MUPIX_VOLTAGEDACS;

#define MUPIX_VOLTAGEDACS_STR(_name) const char *_name[] = {\
"[.]",\
"ThLow = INT : 291",\
"ThPix = INT : 538",\
"BLPix = INT : 538",\
"BLDig = INT : 248",\
"ThHigh = INT : 297",\
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
