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
  INT dummy_data_n;
  BOOL dummy_data_fast;
  BOOL prbs_decode_bypass;
  BOOL reset_datapath;
  BOOL reset_asics;
  BOOL reset_boards;
  BOOL mask[128];
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
    {"VNTimerDel", 20},
    {"VPTimerDel", 1},
    {"VNDAC", 0},
    {"VPFoll", 20},
    {"VNComp", 0},
    {"VNHB", 63},
    {"VPComp2", 5},
    {"VPPump", 63},
    {"VNLVDSDel", 0},
    {"VNLVDS", 16},
    {"VNDcl", 15},
    {"VPDcl", 30},
    {"VNDelPreEmp", 32},
    {"VPDelPreEmp", 32},
    {"VNDelDcl", 32},
    {"VPDelDcl", 32},
    {"VNDelDclMux", 32},
    {"VPDelDclMux", 32},
    {"VNVCO", 23},
    {"VPVCO", 22},
    {"VNOutPix", 10},
    {"VPLoadPix", 10},
    {"VNBiasPix", 0},
    {"BLResDig", 5},
    {"VNPix2", 0},
    {"VPDAC", 0},
    {"VPComp1", 0},
    {"VNDel", 10},
    {"VNRegC", 0},
    {"VNFollPix", 12},
    {"VNFBPix", 4},
    {"VNPix", 20},
    {"ThRes", 0},
    {"BLResPix", 5},
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
    {"TestOut", 0},
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
    {"timerend", 2},
    {"ckdivend2", 31},
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
    {"BLPix", 114},
    {"ThPix", 0},
    {"ThHigh", 0},
    {"ThLow", 82},
    {"ThHigh2", 0},
    {"ThLow2", 0},
    {"Baseline", 70},
    {"VDAC1", 0},
    {"ref_Vss", 184}
};

#endif

#ifndef MUPIX_PSLL_DEFINED
#define MUPIX_PSLL_DEFINED

//// PSLL
/// TODO: we need this for different SWBs in the future for now its only central
constexpr uint32_t per_fe_PSLL_size = 4;
constexpr uint32_t lvds_links_per_feb = 36;
const std::string banknamePSLL = "PSLL";
const std::string namestrPSLL = "Names PSLL";

#endif

#endif //MU3EDAQ_MUPIX_MIDAS_CONFIG_H
