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
#include <array>
using midas::odb;

typedef unsigned int DWORD;
typedef DWORD BOOL;
typedef int INT;

//TODO: The fine grained inclusion giards here actually dangerous and allow for these constants
// to be multiply defined, which produces very nice bugs. Get rid of them!


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
        //{"Num asics", 6},   // This is for the EDM 122021 run
        //{"Num boards", 2},  // This is for the EDM 122021 run
        {"Num asics", 120},   // This is for the 2022 cosmic run
        {"Num boards", 10},  // This is for the 2022 cosmic run
        {"Num rows", 250},
        {"Num cols", 256},
};

#endif

#ifndef MUPIX_DAQ_DEFINED
#define MUPIX_DAQ_DEFINED

typedef struct {
  INT default_th_int_run_2021;
  BOOL dummy_config;
  BOOL dummy_data;
  INT dummy_data_n;
  BOOL dummy_data_fast;
  BOOL prbs_decode_bypass;
  BOOL reset_datapath;
  BOOL reset_asics;
  BOOL reset_boards;
  BOOL mask[128];
} MUPIX_DAQ;

static odb MUPIX_DAQ_SETTINGS = {
    {"default_th_int_run_2021", 0x52},
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
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false,
                false, false, false, false
    }},
};

#endif

#ifndef MUPIX_CHIPDACS_DEFINED
#define MUPIX_CHIPDACS_DEFINED

typedef struct {
    INT VNTimerDel;
    INT VPTimerDel;
    INT VNDAC;
    INT VPFoll;
    INT VNComp;
    INT VNHB;
    INT VPComp2;
    INT VPPump;
    INT VNLVDSDel;
    INT VNLVDS;
    INT VNDcl;
    INT VPDcl;
    INT VNDelPreEmp;
    INT VPDelPreEmp;
    INT VNDelDcl;
    INT VPDelDcl;
    INT VNDelDclMux;
    INT VPDelDclMux;
    INT VNVCO;
    INT VPVCO;
    INT VNOutPix;
    INT VPLoadPix;
    INT VNBiasPix;
    INT BLResDig;
    INT VNPix2;
    INT VPDAC;
    INT VPComp1;
    INT VNDel;
    INT VNRegC;
    INT VNFollPix;
    INT VNFBPix;
    INT VNPix;
    INT ThRes;
    INT BLResPix;
    INT BiasBlock_on;
    INT Bandgap_on;
} MUPIX_BIASDACS;

static odb MUPIX_BIASDACS_SETTINGS = {
    {"VNTimerDel", 40},
    {"VPTimerDel", 1},
    {"VNDAC", 0},
    {"VPFoll", 20},
    {"VNComp", 0},
    {"VNHB", 20},
    {"VPComp2", 10},
    {"VPPump", 63},
    {"VNLVDSDel", 0},
    {"VNLVDS", 10},
    {"VNDcl", 15},
    {"VPDcl", 30},
    {"VNDelPreEmp", 10},
    {"VPDelPreEmp", 10},
    {"VNDelDcl", 5},
    {"VPDelDcl", 5},
    {"VNDelDclMux", 10},
    {"VPDelDclMux", 10},
    {"VNVCO", 13},
    {"VPVCO", 12},
    {"VNOutPix", 5},
    {"VPLoadPix", 2},
    {"VNBiasPix", 0},
    {"BLResDig", 2},
    {"VNPix2", 0},
    {"VPDAC", 0},
    {"VPComp1", 10},
    {"VNDel", 10},
    {"VNRegC", 0},
    {"VNFollPix", 2},
    {"VNFBPix", 5},
    {"VNPix", 10},
    {"ThRes", 0},
    {"BLResPix", 2},
    {"BiasBlock_on", 5},
    {"Bandgap_on", 0}
};

typedef struct {
    INT SelFast;
    INT count_sheep;
    INT NC1;
    INT TestOut;
    INT disable_HB;
    INT conf_res_n;
    INT RO_res_n;
    INT Ser_res_n;
    INT Aur_res_n;
    INT NC2;
    INT Tune_Reg_L;
    INT NC3;
    INT Tune_Reg_R;
    INT AlwaysEnable;
    INT En2thre;
    INT NC4;
    INT EnPLL;
    INT SelSlow;
    INT SelEx;
    INT invert;
    INT slowdownlDColEnd;
    INT EnSync_SC;
    INT NC5;
    INT linksel;
    INT tsphase;
    INT sendcounter;
    INT resetckdivend;
    INT NC6;
    INT maxcycend;
    INT slowdownend;
    INT timerend;
    INT ckdivend2;
    INT ckdivend;
} MUPIX_CONFDACS;

static odb MUPIX_CONFDACS_SETTINGS = {
    {"SelFast", 0},
    {"count_sheep", 0},
    {"NC1", 0},
    {"TestOut", 1},
    {"disable_HB", 1},
    {"conf_res_n", 1},
    {"RO_res_n", 1},
    {"Ser_res_n", 1},
    {"Aur_res_n", 1},
    {"NC2", 0},
    {"Tune_Reg_L", 0},
    {"NC3", 0},
    {"Tune_Reg_R", 0},
    {"AlwaysEnable", 1},
    {"En2thre", 0},
    {"NC4", 0},
    {"EnPLL", 0},
    {"SelSlow", 0},
    {"SelEx", 0},
    {"invert", 0},
    {"slowdownlDColEnd", 7},
    {"EnSync_SC", 0},
    {"NC5", 0},
    {"linksel", 0},
    {"tsphase", 0},
    {"sendcounter", 0},
    {"resetckdivend", 0},
    {"NC6", 0},
    {"maxcycend", 63},
    {"slowdownend", 0},
    {"timerend", 1},
    {"ckdivend2", 15},
    {"ckdivend", 0}
};

typedef struct {
    INT VCAL;
    INT BLPix;
    INT ThPix;
    INT ThHigh;
    INT ThLow;
    INT ThHigh2;
    INT ThLow2;
    INT Baseline;
    INT VDAC1;
    INT ref_Vss;
} MUPIX_VDACS;

static odb MUPIX_VDACS_SETTINGS = {
    {"VCAL", 0},
    {"BLPix", 151},
    {"ThPix", 0},
    {"ThHigh", 185},
    {"ThLow", 181},
    {"ThHigh2", 0},
    {"ThLow2", 0},
    {"Baseline", 167},
    {"VDAC1", 0},
    {"ref_Vss", 198}
};

#endif

#ifndef MUPIX_TDACS_DEFINED
#define MUPIX_TDACS_DEFINED

typedef struct {
    std::string TDACFILE;
} MUPIX_TDACS;

static odb MUPIX_TDACS_SETTINGS = {
    {"TDACFILE", "default_tdacs_mupix.csv"}
};

#endif

#ifndef MUPIX_FEBS_DEFINED
#define MUPIX_FEBS_DEFINED

typedef struct {
    INT MP_LVDS_LINK_MASK;
    INT MP_LVDS_LINK_MASK2;
} MUPIX_FEBS;

static odb MUPIX_FEB_SETTINGS = {
    {"MP_LVDS_LINK_MASK", 0x0},
    {"MP_LVDS_LINK_MASK2", 0x0},
};

#endif

#ifndef MUPIX_GLOBAL_FEBS_DEFINED
#define MUPIX_GLOBAL_FEBS_DEFINED

typedef struct {
    INT ASICsPerFEB;
} MUPIX_GLOBAL_FEBS;

static odb MUPIX_GLOBAL_FEBS_SETTINGS = {
    //{"ASICsPerFEB", 3}  // This is for the EDM 122021 run
    {"ASICsPerFEB", 12}  // This is for the CosmicRun 042022 run
};

#endif




#endif //MU3EDAQ_MUPIX_MIDAS_CONFIG_H
