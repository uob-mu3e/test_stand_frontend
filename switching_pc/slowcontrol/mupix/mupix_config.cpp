#include <cstring>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include "mupix_config.h"

namespace mupix {

/// MUPIX board configuration

MupixBoardConfig::paras_t MupixBoardConfig::parameters_boarddacs = {
    make_param("Threshold_Low", 16, true),
    make_param("Threshold_High", 16, true),
    make_param("Injection", 16, true),
    make_param("Threshold_Pix", 16, true),
    make_param("TDiode_Current", 16, true),
    make_param("TDiode_ADC", 16, true)
};

MupixBoardConfig::MupixBoardConfig() {
    // populate name/offset map
    length_bits = 0;
    // header 
    for(const auto& para : parameters_boarddacs)
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

MupixBoardConfig::~MupixBoardConfig() {
    delete[] bitpattern_r;
    delete[] bitpattern_w;
}

void MupixBoardConfig::Parse_BoardDACs_from_struct(MUPIX_BOARDDACS mt_g){
    setParameter("Threshold_High", mt_g.Threshold_High);
    setParameter("Threshold_Low", mt_g.Threshold_Low);
    setParameter("Threshold_Pix", mt_g.Threshold_Pix);
    setParameter("Injection", mt_g.Injection);
    setParameter("TDiode_ADC", mt_g.TDiode_ADC);
}

/// MUPIX configuration
// Pattern is assembled based on the scheme described in the mupix8 documentation, section 10

MupixConfig::paras_t MupixConfig::parameters_bias = {
    make_param("VNTimerDel", "5 4 3 2 1 0"),
    make_param("VPTimerDel", "5 4 3 2 1 0"),
    make_param("VNDAC", "5 4 3 2 1 0"),
    make_param("VPFoll", "5 4 3 2 1 0"),
    make_param("VNComp", "5 4 3 2 1 0"),
    make_param("VNHB", "5 4 3 2 1 0"),
    make_param("VPComp2", "5 4 3 2 1 0"),
    make_param("VPPump", "5 4 3 2 1 0"),
    make_param("VNLVDSDel", "5 4 3 2 1 0"),
    make_param("VNLVDS", "5 4 3 2 1 0"),
    make_param("VNDcl", "5 4 3 2 1 0"),
    make_param("VPDcl", "5 4 3 2 1 0"),
    make_param("VNDelPreEmp", "5 4 3 2 1 0"),
    make_param("VPDelPreEmp", "5 4 3 2 1 0"),
    make_param("VNDelDcl", "5 4 3 2 1 0"),
    make_param("VPDelDcl", "5 4 3 2 1 0"),
    make_param("VNDelDclMux", "5 4 3 2 1 0"),
    make_param("VPDelDclMux", "5 4 3 2 1 0"),
    make_param("VNVCO", "5 4 3 2 1 0"),
    make_param("VPVCO", "5 4 3 2 1 0"),
    make_param("VNOutPix", "5 4 3 2 1 0"),
    make_param("VPLoadPix", "5 4 3 2 1 0"),
    make_param("VNBiasPix", "5 4 3 2 1 0"),
    make_param("BLResDig", "5 4 3 2 1 0"),
    make_param("VNPix2", "5 4 3 2 1 0"),
    make_param("VPDAC", "5 4 3 2 1 0"),
    make_param("VPComp1", "5 4 3 2 1 0"),
    make_param("VNDel", "5 4 3 2 1 0"),
    make_param("VNRegC", "5 4 3 2 1 0"),
    make_param("VNFollPix", "5 4 3 2 1 0"),
    make_param("VNFBPix", "5 4 3 2 1 0"),
    make_param("VNPix", "5 4 3 2 1 0"),
    make_param("ThRes", "5 4 3 2 1 0"),
    make_param("BLResPix", "5 4 3 2 1 0"),
    make_param("BiasBlock_on", "2 1 0"),
    make_param("Bandgap_on", "0")
};

MupixConfig::paras_t MupixConfig::parameters_conf = {
    make_param("SelFast", "0"),
    make_param("count_sheep", "0"),
    make_param("NC1", "0 1 2 3 4"),
    make_param("TestOut", "0 1 2 3"),
    make_param("disable_HB", "0"),
    make_param("conf_res_n", "0"),
    make_param("RO_res_n", "0"),
    make_param("Ser_res_n", "0"),
    make_param("Aur_res_n", "0"),
    make_param("NC2", "0"),
    make_param("Tune_Reg_L", "0 1 2 3 4 5"),
    make_param("NC3", "0"),
    make_param("Tune_Reg_R", "0 1 2 3 4 5"),
    make_param("AlwaysEnable", "0"),
    make_param("En2thre", "0"),
    make_param("NC4", "0 1 2 3"),
    make_param("EnPLL", "0"),
    make_param("SelSlow", "0"),
    make_param("SelEx", "0"),
    make_param("invert", "0"),
    make_param("slowdownlDColEnd", "0 1 2 3 4"),
    make_param("EnSync_SC", "0"),
    make_param("NC5", "0 1 2"),
    make_param("linksel", "0 1"),
    make_param("tsphase", "0 1 2 3 4 5"),
    make_param("sendcounter", "0"),
    make_param("resetckdivend", "0 1 2 3"),
    make_param("NC6", "0 1"),
    make_param("maxcycend", "0 1 2 3 4 5"),
    make_param("slowdownend", "0 1 2 3"),
    make_param("timerend", "0 1 2 3"),
    make_param("ckdivend2", "0 1 2 3 4 5"),
    make_param("ckdivend", "0 1 2 3 4 5")
};

MupixConfig::paras_t MupixConfig::parameters_vdacs = {
    make_param("VCAL", "0 1 2 3 4 5 6 7"),
    make_param("BLPix", "0 1 2 3 4 5 6 7"),
    make_param("ThPix", "0 1 2 3 4 5 6 7"),
    make_param("ThHigh", "0 1 2 3 4 5 6 7"),
    make_param("ThLow", "0 1 2 3 4 5 6 7"),
    make_param("ThHigh2", "0 1 2 3 4 5 6 7"),
    make_param("ThLow2", "0 1 2 3 4 5 6 7"),
    make_param("Baseline", "0 1 2 3 4 5 6 7"),
    make_param("VDAC1", "0 1 2 3 4 5 6 7"),
    make_param("ref_Vss", "0 1 2 3 4 5 6 7")
};


MupixConfig::MupixConfig() {
    length_bits = 0;

    for(const auto& para : MupixConfig::parameters_bias)
        addPara(para, "");
    for(const auto& para : MupixConfig::parameters_conf)
        addPara(para, "");
    for(const auto& para : MupixConfig::parameters_vdacs)
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

MupixConfig::~MupixConfig() {
}

void MupixConfig::Parse_BiasDACs_from_struct(MUPIX_BIASDACS mt) {
    setParameter("VNTimerDel", mt.VNTimerDel, true);
    setParameter("VPTimerDel", mt.VPTimerDel, true);
    setParameter("VNDAC", mt.VNDAC, true);
    setParameter("VPFoll", mt.VPFoll, true);
    setParameter("VNComp", mt.VNComp, true);
    setParameter("VNHB", mt.VNHB, true);
    setParameter("VPComp2", mt.VPComp2, true);
    setParameter("VPPump", mt.VPPump, true);
    setParameter("VNLVDSDel", mt.VNLVDSDel, true);
    setParameter("VNLVDS", mt.VNLVDS, true);
    setParameter("VNDcl", mt.VNDcl, true);
    setParameter("VPDcl", mt.VPDcl, true);
    setParameter("VNDelPreEmp", mt.VNDelPreEmp, true);
    setParameter("VPDelPreEmp", mt.VPDelPreEmp, true);
    setParameter("VNDelDcl", mt.VNDelDcl, true);
    setParameter("VPDelDcl", mt.VPDelDcl, true);
    setParameter("VNDelDclMux", mt.VNDelDclMux, true);
    setParameter("VPDelDclMux", mt.VPDelDclMux, true);
    setParameter("VNVCO", mt.VNVCO, true);
    setParameter("VPVCO", mt.VPVCO, true);
    setParameter("VNOutPix", mt.VNOutPix, true);
    setParameter("VPLoadPix", mt.VPLoadPix, true);
    setParameter("VNBiasPix", mt.VNBiasPix, true);
    setParameter("BLResDig", mt.BLResDig, true);
    setParameter("VNPix2", mt.VNPix2, true);
    setParameter("VPDAC", mt.VPDAC, true);
    setParameter("VPComp1", mt.VPComp1, true);
    setParameter("VNDel", mt.VNDel, true);
    setParameter("VNRegC", mt.VNRegC, true);
    setParameter("VNFollPix", mt.VNFollPix, true);
    setParameter("VNFBPix", mt.VNFBPix, true);
    setParameter("VNPix", mt.VNPix, true);
    setParameter("ThRes", mt.ThRes, true);
    setParameter("BLResPix", mt.BLResPix, true);
    setParameter("BiasBlock_on", mt.BiasBlock_on, true);
    setParameter("Bandgap_on", mt.Bandgap_on, true);
}

void MupixConfig::Parse_ConfDACs_from_struct(MUPIX_CONFDACS mt) {
    setParameter("SelFast", mt.SelFast, true);
    setParameter("count_sheep", mt.count_sheep, true);
    setParameter("TestOut", mt.TestOut, true);
    setParameter("disable_HB", mt.disable_HB, true);
    setParameter("conf_res_n", mt.conf_res_n, true);
    setParameter("RO_res_n", mt.RO_res_n, true);
    setParameter("Ser_res_n", mt.Ser_res_n, true);
    setParameter("Aur_res_n", mt.Aur_res_n, true);
    setParameter("Tune_Reg_L", mt.Tune_Reg_L, true);
    setParameter("Tune_Reg_R", mt.Tune_Reg_R, true);
    setParameter("AlwaysEnable", mt.AlwaysEnable, true);
    setParameter("En2thre", mt.En2thre, true);
    setParameter("EnPLL", mt.EnPLL, true);
    setParameter("SelSlow", mt.SelSlow, true);
    setParameter("SelEx", mt.SelEx, true);
    setParameter("invert", mt.invert, true);
    setParameter("slowdownlDColEnd", mt.slowdownlDColEnd, true);
    setParameter("EnSync_SC", mt.EnSync_SC, true);
    setParameter("linksel", mt.linksel, true);
    setParameter("tsphase", mt.tsphase, true);
    setParameter("sendcounter", mt.sendcounter, true);
    setParameter("resetckdivend", mt.resetckdivend, true);
    setParameter("maxcycend", mt.maxcycend, true);
    setParameter("slowdownend", mt.slowdownend, true);
    setParameter("timerend", mt.timerend, true);
    setParameter("ckdivend2", mt.ckdivend2, true);
    setParameter("ckdivend", mt.ckdivend, true);
    setParameter("NC1", mt.NC1, true);
    setParameter("NC2", mt.NC2, true);
    setParameter("NC3", mt.NC3, true);
    setParameter("NC4", mt.NC4, true);
    setParameter("NC5", mt.NC5, true);
    setParameter("NC6", mt.NC6, true);
}

void MupixConfig::Parse_VDACs_from_struct(MUPIX_VDACS mt) {
    setParameter("VCAL", mt.VCAL, true);
    setParameter("BLPix", mt.BLPix, true);
    setParameter("ThPix", mt.ThPix, true);
    setParameter("ThHigh", mt.ThHigh, true);
    setParameter("ThLow", mt.ThLow, true);
    setParameter("ThHigh2", mt.ThHigh2, true);
    setParameter("ThLow2", mt.ThLow2, true);
    setParameter("Baseline", mt.Baseline, true);
    setParameter("VDAC1", mt.VDAC1, true);
    setParameter("ref_Vss", mt.ref_Vss, true);
}

} // namespace mutrig
