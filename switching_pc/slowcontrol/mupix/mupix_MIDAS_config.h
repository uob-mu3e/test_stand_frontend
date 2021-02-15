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
#include <odbxx.h>
using midas::odb;

typedef unsigned int DWORD;
typedef DWORD BOOL;
typedef int INT;


#ifndef MU3EDAQ_MUPIX_MIDAS_CONFIG_H
#define MU3EDAQ_MUPIX_MIDAS_CONFIG_H

#ifndef MUPIX_GLOBAL_DEFINED
#define MUPIX_GLOBAL_DEFINED

typedef struct {
    INT       n_asics;
    INT       n_boards;
    INT       n_rows;
    INT       n_cols;
} MUPIX_GLOBAL;

static odb MUPIX_GLOBAL_SETTINGS = {
        {"Num asics", 0},
        {"Num boards", 0},
        {"Num rows", 200},
        {"Num cols", 128},
};

#endif

#ifndef MUPIX_DAQ_DEFINED
#define MUPIX_DAQ_DEFINED

typedef struct {
  BOOL dummy_config;
  BOOL dummy_data;
} MUPIX_DAQ;

static odb MUPIX_DAQ_SETTINGS = {
    {"dummy_config", false},
    {"dummy_data", false},
    {"dummy_data_n", 255},
    {"dummy_data_fast", false},
    {"prbs_decode_bypass", false},
    {"reset_datapath", false},
    {"reset_asics", false},
    {"reset_boards", false},
    {"mask", {  false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false
    }},
};

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

static odb MUPIX_CHIPDACS_SETTINGS = {
    {"Bandgap1_on", 0},
    {"Biasblock1_on", 5},
    {"unused_1", 0},
    {"unused_2", 0},
    {"unused_3", 0},
    {"unused_4", 0},
    {"unused_5", 0},
    {"VNRegCasc", 0},
    {"VDel", 16},
    {"VPComp", 5},
    {"VPDAC", 0},
    {"unused_6", 0},
    {"BLResDig", 5},
    {"unused_7", 0},
    {"unused_8", 0},
    {"unused_9", 0},
    {"VPVCO", 12},
    {"VNVCO", 13},
    {"VPDelDclMux", 24},
    {"VNDelDclMux", 24},
    {"VPDelDcl", 40},
    {"VNDelDcl", 40},
    {"VPDelPreEmp", 24},
    {"VNDelPreEmp", 24},
    {"VPDcl", 24},
    {"VNDcl", 16},
    {"VNLVDS", 24},
    {"VNLVDSDel", 0},
    {"VPPump", 63},
    {"resetckdivend", 15},
    {"maxcycend", 63},
    {"slowdownend", 0},
    {"timerend", 1},
    {"tsphase", 0},
    {"ckdivend2", 7},
    {"ckdivend", 0},
    {"VPRegCasc", 0},
    {"VPRamp", 0},
    {"unused_10", 0},
    {"unused_11", 0},
    {"unused_12", 0},
    {"VPBiasReg", 0},
    {"VNBiasReg", 0},
    {"enable2threshold", 0},
    {"enableADC", 1},
    {"Invert", 0},
    {"SelEx", 0},
    {"SelSlow", 0},
    {"EnablePLL", 1},
    {"Readout_reset_n", 1},
    {"Serializer_reset_n", 1},
    {"Aurora_reset_n", 1},
    {"sendcounter", 0},
    {"Linkselect", 1},
    {"Termination", 0},
    {"AlwaysEnable", 0},
    {"SelectTest", 0},
    {"SelectTestOut", 0},
    {"DisableHitbus", 1},
    {"unused_13", 0},
};

typedef  struct {
    int       digiWrite;
} MUPIX_DIGIROWDACS;

static odb MUPIX_DIGIROWDACS_SETTINGS = {
    {"digiWrite", 0}, 
};

typedef  struct {
    int       RAM;
    int       EnableHitbus;
    int       EnableInjection;
} MUPIX_COLDACS;

static odb MUPIX_COLDACS_SETTINGS = {
    {"RAM", 0}, 
    {"EnableHitbus", 0}, 
    {"EnableInjection", 0},
};

typedef  struct {
    int       unused;
    int       EnableInjection;
    int       EnableAnalogueBuffer;
} MUPIX_ROWDACS;

static odb MUPIX_ROWDACS_SETTINGS = {
    {"unused", 0}, 
    {"EnableInjection", 0}, 
    {"EnableAnalogueBuffer", 0},
};

typedef struct {
    int       Bandgap2_on;
    int       Biasblock2_on;
    int       BLResPix;
    int       unused_14;
    int       VNPix;
    int       VNFBPix;
    int       VNFollPix;
    int       unused_15;
    int       unused_16;
    int       unused_17;
    int       unused_18;
    int       VNPix2;
    int       unused_19;
    int       VNBiasPix;
    int       VPLoadPix;
    int       VNOutPix;
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
    int       VPFoll;
    int       VNDACPix;
    int       unused_43;
    int       unused_44;
    int       unused_45;
    int       unused_46;
    int       unused_47;
    int       unused_48;
    int       unused_49;
    int       unused_50;
    int       unused_51;
    int       unused_52;
    int       unused_53;
    int       unused_54;
    int       unused_55;
    int       unused_56;
    int       unused_57;
    int       unused_58;
    int       unused_59;
    int       unused_60;
    int       unused_61;
} MUPIX_CHIPDACS2;

static odb MUPIX_CHIPDACS2_SETTINGS = {
    {"Bandgap2_on", 0},
    {"Biasblock2_on", 5},
    {"BLResPix", 5},
    {"unused_14", 0},
    {"VNPix", 20},
    {"VNFBPix", 10},
    {"VNFollPix", 10},
    {"unused_15", 0},
    {"unused_16", 0},
    {"unused_17", 0},
    {"unused_18", 0},
    {"VNPix2", 0},
    {"unused_19", 0},
    {"VNBiasPix", 0},
    {"VPLoadPix", 5},
    {"VNOutPix", 10},
    {"unused_20", 0},
    {"unused_21", 0},
    {"unused_22", 0},
    {"unused_23", 0},
    {"unused_24", 0},
    {"unused_25", 0},
    {"unused_26", 0},
    {"unused_27", 0},
    {"unused_28", 0},
    {"unused_29", 0},
    {"unused_30", 0},
    {"unused_31", 0},
    {"unused_32", 0},
    {"unused_33", 0},
    {"unused_34", 0},
    {"unused_35", 0},
    {"unused_36", 0},
    {"unused_37", 0},
    {"unused_38", 0},
    {"unused_39", 0},
    {"unused_40", 0},
    {"unused_41", 0},
    {"unused_42", 0},
    {"VPFoll", 10},
    {"VNDACPix", 0},
    {"unused_43", 0},
    {"unused_44", 0},
    {"unused_45", 0},
    {"unused_46", 0},
    {"unused_47", 0},
    {"unused_48", 0},
    {"unused_49", 0},
    {"unused_50", 0},
    {"unused_51", 0},
    {"unused_52", 0},
    {"unused_53", 0},
    {"unused_54", 0},
    {"unused_55", 0},
    {"unused_56", 0},
    {"unused_57", 0},
    {"unused_58", 0},
    {"unused_59", 0},
    {"unused_60", 0},
    {"unused_61", 0},
};

typedef struct {
    int       ThLow;
    int       ThPix;
    int       BLPix;
    int       BLDig;
    int       ThHigh;
} MUPIX_VOLTAGEDACS;

static odb MUPIX_VOLTAGEDACS_SETTINGS = {
    {"ThLow", 291},
    {"ThPix", 538},
    {"BLPix", 538},
    {"BLDig", 248},
    {"ThHigh", 297},
};

typedef struct {
    INT RAM[128];
    BOOL EnableHitbus[128];
    BOOL EnableInjection[128];
} MUPIX_PIXELROWCONFIG;


static odb MUPIX_PIXELROWCONFIG_SETTINGS = {
    {"RAM", std::array<int, 128>{}},
    {"EnableHitbus", std::array<int, 128>{}},
    {"EnableInjection", std::array<int, 128>{}},
};

#endif

#ifndef MUPIX_BOARDDACS_DEFINED
#define MUPIX_BOARDDACS_DEFINED

typedef struct {
    INT      Threshold_High;
    INT      Threshold_Low;
    INT      Threshold_Pix;
    INT      Injection;
    INT      TDiode_Current;
    INT      TDiode_ADC;
} MUPIX_BOARDDACS;

static odb MUPIX_BOARDDACS_SETTINGS = {
    {"Threshold_High", 19312},
    {"Threshold_Low", 19312},
    {"Threshold_Pix", 48284},
    {"Injection", 0},
    {"TDiode_Current", 0},
    {"TDiode_ADC", 0},
};

#endif


#endif //MU3EDAQ_MUPIX_MIDAS_CONFIG_H
