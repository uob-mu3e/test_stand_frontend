#include <cstring>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include "mupix_config.h"

namespace mupix {

/// MUPIX configuration

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
    setParameter("VNTimerDel", mt.VNTimerDel, false);
    setParameter("VPTimerDel", mt.VPTimerDel, false);
    setParameter("VNDAC", mt.VNDAC, false);
    setParameter("VPFoll", mt.VPFoll, false);
    setParameter("VNComp", mt.VNComp, false);
    setParameter("VNHB", mt.VNHB, false);
    setParameter("VPComp2", mt.VPComp2, false);
    setParameter("VPPump", mt.VPPump, false);
    setParameter("VNLVDSDel", mt.VNLVDSDel, false);
    setParameter("VNLVDS", mt.VNLVDS, false);
    setParameter("VNDcl", mt.VNDcl, false);
    setParameter("VPDcl", mt.VPDcl, false);
    setParameter("VNDelPreEmp", mt.VNDelPreEmp, false);
    setParameter("VPDelPreEmp", mt.VPDelPreEmp, false);
    setParameter("VNDelDcl", mt.VNDelDcl, false);
    setParameter("VPDelDcl", mt.VPDelDcl, false);
    setParameter("VNDelDclMux", mt.VNDelDclMux, false);
    setParameter("VPDelDclMux", mt.VPDelDclMux, false);
    setParameter("VNVCO", mt.VNVCO, false);
    setParameter("VPVCO", mt.VPVCO, false);
    setParameter("VNOutPix", mt.VNOutPix, false);
    setParameter("VPLoadPix", mt.VPLoadPix, false);
    setParameter("VNBiasPix", mt.VNBiasPix, false);
    setParameter("BLResDig", mt.BLResDig, false);
    setParameter("VNPix2", mt.VNPix2, false);
    setParameter("VPDAC", mt.VPDAC, false);
    setParameter("VPComp1", mt.VPComp1, false);
    setParameter("VNDel", mt.VNDel, false);
    setParameter("VNRegC", mt.VNRegC, false);
    setParameter("VNFollPix", mt.VNFollPix, false);
    setParameter("VNFBPix", mt.VNFBPix, false);
    setParameter("VNPix", mt.VNPix, false);
    setParameter("ThRes", mt.ThRes, false);
    setParameter("BLResPix", mt.BLResPix, false);
    setParameter("BiasBlock_on", mt.BiasBlock_on, false);
    setParameter("Bandgap_on", mt.Bandgap_on, false);
}

void MupixConfig::Parse_ConfDACs_from_struct(MUPIX_CONFDACS mt) {
    setParameter("SelFast", mt.SelFast, false);
    setParameter("count_sheep", mt.count_sheep, false);
    setParameter("TestOut", mt.TestOut, false);
    setParameter("disable_HB", mt.disable_HB, false);
    setParameter("conf_res_n", mt.conf_res_n, false);
    setParameter("RO_res_n", mt.RO_res_n, false);
    setParameter("Ser_res_n", mt.Ser_res_n, false);
    setParameter("Aur_res_n", mt.Aur_res_n, false);
    setParameter("Tune_Reg_L", mt.Tune_Reg_L, false);
    setParameter("Tune_Reg_R", mt.Tune_Reg_R, false);
    setParameter("AlwaysEnable", mt.AlwaysEnable, false);
    setParameter("En2thre", mt.En2thre, false);
    setParameter("EnPLL", mt.EnPLL, false);
    setParameter("SelSlow", mt.SelSlow, false);
    setParameter("SelEx", mt.SelEx, false);
    setParameter("invert", mt.invert, false);
    setParameter("slowdownlDColEnd", mt.slowdownlDColEnd, false);
    setParameter("EnSync_SC", mt.EnSync_SC, false);
    setParameter("linksel", mt.linksel, false);
    setParameter("tsphase", mt.tsphase, false);
    setParameter("sendcounter", mt.sendcounter, false);
    setParameter("resetckdivend", mt.resetckdivend, false);
    setParameter("maxcycend", mt.maxcycend, false);
    setParameter("slowdownend", mt.slowdownend, false);
    setParameter("timerend", mt.timerend, false);
    setParameter("ckdivend2", mt.ckdivend2, false);
    setParameter("ckdivend", mt.ckdivend, false);
    setParameter("NC1", mt.NC1, false);
    setParameter("NC2", mt.NC2, false);
    setParameter("NC3", mt.NC3, false);
    setParameter("NC4", mt.NC4, false);
    setParameter("NC5", mt.NC5, false);
    setParameter("NC6", mt.NC6, false);
}

void MupixConfig::Parse_VDACs_from_struct(MUPIX_VDACS mt) {
    setParameter("VCAL", mt.VCAL, false);
    setParameter("BLPix", mt.BLPix, false);
    setParameter("ThPix", mt.ThPix, false);
    setParameter("ThHigh", mt.ThHigh, false);
    setParameter("ThLow", mt.ThLow, false);
    setParameter("ThHigh2", mt.ThHigh2, false);
    setParameter("ThLow2", mt.ThLow2, false);
    setParameter("Baseline", mt.Baseline, false);
    setParameter("VDAC1", mt.VDAC1, false);
    setParameter("ref_Vss", mt.ref_Vss, false);
}

} // namespace mutrig
