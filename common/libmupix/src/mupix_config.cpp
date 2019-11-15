#include <cstring>
#include <iostream>
#include <iomanip>

#include "mupix_config.h"

namespace mupix {


/// MUPIX configuration
// Pattern is assembled based on the scheme described in the mupix8 documentation, section 10

MupixConfig::paras_t MupixConfig::parameters_chipdacs = {
    make_param("Bandgap1_on", "0 "),
    make_param("Biasblock1_on", "0 1 2 "),
    make_param("unused_1", "2 0 1 3 4 5 "),
    make_param("unused_2", "2 0 1 3 4 5 "),
    make_param("unused_3", "2 0 1 3 4 5 "),
    make_param("unused_4", "2 0 1 3 4 5 "),
    make_param("unused_5", "2 0 1 3 4 5 "),
    make_param("VNRegCasc", "2 0 1 3 4 5 "),
    make_param("VDel", "2 0 1 3 4 5 "),
    make_param("VPComp", "2 0 1 3 4 5 "),
    make_param("VPDAC", "2 0 1 3 4 5 "),
    make_param("unused_6", "2 0 1 3 4 5 "),
    make_param("BLResDig", "2 0 1 3 4 5 "),
    make_param("unused_7", "2 0 1 3 4 5 "),
    make_param("unused_8", "2 0 1 3 4 5 "),
    make_param("unused_9", "2 0 1 3 4 5 "),
    make_param("VPVCO", "2 0 1 3 4 5 "),
    make_param("VNVCO", "2 0 1 3 4 5 "),
    make_param("VPDelDclMux", "2 0 1 3 4 5 "),
    make_param("VNDelDclMux", "2 0 1 3 4 5 "),
    make_param("VPDelDcl", "2 0 1 3 4 5 "),
    make_param("VNDelDcl", "2 0 1 3 4 5 "),
    make_param("VPDelPreEmp", "2 0 1 3 4 5 "),
    make_param("VNDelPreEmp", "2 0 1 3 4 5 "),
    make_param("VPDcl", "2 0 1 3 4 5 "),
    make_param("VNDcl", "2 0 1 3 4 5 "),
    make_param("VNLVDS", "2 0 1 3 4 5 "),
    make_param("VNLVDSDel", "2 0 1 3 4 5 "),
    make_param("VPPump", "2 0 1 3 4 5 "),
    make_param("resetckdivend", "0 1 2 3 "),
    make_param("maxcycend", "0 1 2 3 4 5 "),
    make_param("slowdownend", "0 1 2 3 "),
    make_param("timerend", "0 1 2 3 "),
    make_param("tsphase", "0 1 2 3 4 5 "),
    make_param("ckdivend2", "0 1 2 3 4 5 "),
    make_param("ckdivend", "0 1 2 3 4 5 "),
    make_param("VPRegCasc", "2 0 1 3 4 5 "),
    make_param("VPRamp", "2 0 1 3 4 5 "),
    make_param("unused_10", "2 0 1 3 4 5 "),
    make_param("unused_11", "2 0 1 3 4 5 "),
    make_param("unused_12", "2 0 1 3 4 5 "),
    make_param("VPBiasReg", "2 0 1 3 4 5 "),
    make_param("VNBiasReg", "2 0 1 3 4 5 "),
    make_param("enable2threshold", "0 "),
    make_param("enableADC", "0 "),
    make_param("Invert", "0 "),
    make_param("SelEx", "0 "),
    make_param("SelSlow", "0 "),
    make_param("EnablePLL", "0 "),
    make_param("Readout_reset_n", "0 "),
    make_param("Serializer_reset_n", "0 "),
    make_param("Aurora_reset_n", "0 "),
    make_param("sendcounter", "0 "),
    make_param("Linkselect", "0 1 "),
    make_param("Termination", "0 "),
    make_param("AlwaysEnable", "0 "),
    make_param("SelectTest", "0 "),
    make_param("SelectTestOut", "0 "),
    make_param("DisableHitbus", "0 "),
    make_param("unused_13", "0 "),
};

MupixConfig::paras_t MupixConfig::parameters_digirowdacs = {
    make_param("digiWrite", 1, 0),
};

MupixConfig::paras_t MupixConfig::parameters_coldacs = {
    make_param("RAM", 6, {5,4,3,0,1,2}),
    make_param("EnableHitbus", 1, 0),
    make_param("EnableInjection", 1, 0),
};

MupixConfig::paras_t MupixConfig::parameters_rowdacs = {
    make_param("unused", 4, {3,2,1,0}),
    make_param("EnableInjection", 1, 0),
    make_param("EnableAnalogueBuffer", 1, 0),
};

MupixConfig::paras_t MupixConfig::parameters_chipdacs2 = {
    make_param("Bandgap2_on", "0 "),
    make_param("Biasblock2_on", "0 1 2 "),
    make_param("BLResPix", "2 0 1 3 4 5 "),
    make_param("unused_1", "2 0 1 3 4 5 "),
    make_param("VNPix", "2 0 1 3 4 5 "),
    make_param("VNFBPix", "2 0 1 3 4 5 "),
    make_param("VNFollPix", "2 0 1 3 4 5 "),
    make_param("unused_2", "2 0 1 3 4 5 "),
    make_param("unused_3", "2 0 1 3 4 5 "),
    make_param("unused_4", "2 0 1 3 4 5 "),
    make_param("unused_5", "2 0 1 3 4 5 "),
    make_param("VNPix2", "2 0 1 3 4 5 "),
    make_param("unused_6", "2 0 1 3 4 5 "),
    make_param("VNBiasPix", "2 0 1 3 4 5 "),
    make_param("VPLoadPix", "2 0 1 3 4 5 "),
    make_param("VNOutPix", "2 0 1 3 4 5 "),
    make_param("unused_7", "2 0 1 3 4 5 "),
    make_param("unused_8", "2 0 1 3 4 5 "),
    make_param("unused_9", "2 0 1 3 4 5 "),
    make_param("unused_10", "2 0 1 3 4 5 "),
    make_param("unused_11", "2 0 1 3 4 5 "),
    make_param("unused_12", "2 0 1 3 4 5 "),
    make_param("unused_13", "2 0 1 3 4 5 "),
    make_param("unused_14", "2 0 1 3 4 5 "),
    make_param("unused_15", "2 0 1 3 4 5 "),
    make_param("unused_16", "2 0 1 3 4 5 "),
    make_param("unused_17", "2 0 1 3 4 5 "),
    make_param("unused_18", "2 0 1 3 4 5 "),
    make_param("unused_19", "2 0 1 3 4 5 "),
    make_param("unused_20", "0 1 2 3 "),
    make_param("unused_21", "0 1 2 3 4 5 "),
    make_param("unused_22", "0 1 2 3 "),
    make_param("unused_23", "0 1 2 3 "),
    make_param("unused_24", "0 1 2 3 4 5 "),
    make_param("unused_25", "0 1 2 3 4 5 "),
    make_param("unused_26", "0 1 2 3 4 5 "),
    make_param("unused_27", "2 0 1 3 4 5 "),
    make_param("unused_28", "2 0 1 3 4 5 "),
    make_param("unused_29", "2 0 1 3 4 5 "),
    make_param("VPFoll", "2 0 1 3 4 5 "),
    make_param("VNDACPix", "2 0 1 3 4 5 "),
    make_param("unused_30", "2 0 1 3 4 5 "),
    make_param("unused_31", "2 0 1 3 4 5 "),
    make_param("unused_32", "0 "),
    make_param("unused_33", "0 "),
    make_param("unused_34", "0 "),
    make_param("unused_35", "0 "),
    make_param("unused_36", "0 "),
    make_param("unused_37", "0 "),
    make_param("unused_38", "0 "),
    make_param("unused_39", "0 "),
    make_param("unused_40", "0 "),
    make_param("unused_41", "0 "),
    make_param("unused_42", "0 1 "),
    make_param("unused_43", "0 "),
    make_param("unused_44", "0 "),
    make_param("unused_45", "0 "),
    make_param("unused_46", "0 "),
    make_param("unused_47", "0 "),
    make_param("unused_48", "0 "),};

MupixConfig::paras_t MupixConfig::parameters_voltagedacs = {
    make_param("ThLow", "9 8 7 6 5 4 3 2 1 0 "),
    make_param("ThPix", "9 8 7 6 5 4 3 2 1 0 "),
    make_param("BLPix", "9 8 7 6 5 4 3 2 1 0 "),
    make_param("BLDig", "9 8 7 6 5 4 3 2 1 0 "),
    make_param("ThHigh", "9 8 7 6 5 4 3 2 1 0 ")
};

MupixConfig::MupixConfig() {
    unsigned int nrow = 200;
    unsigned int ncol = 128;
    // populate name/offset map
    length_bits = 0;
    // MUPIX DAC block 1 (10.1)
    for(const auto& para : parameters_chipdacs)
        addPara(para, "");
    for(unsigned int row = 0; row < nrow; ++row)
        for(const auto& para : parameters_digirowdacs)
            addPara(para, "_row"+std::to_string(row));
    // row selection register (write bits, 10.2)
    //addPara(make_param("rowsel_reg",nrow,0), "");
    // pixel row block (pixel settings, 10.3).
    for(unsigned int col = 0; col < ncol; ++col) 
        for(const auto& para : parameters_coldacs )
            addPara(para, "_col"+std::to_string(col));
    // row register (Q bits, 10.4 )
    for(unsigned int row = 0; row < nrow; ++row)
        for(const auto& para : parameters_rowdacs)
            addPara(para, "_row"+std::to_string(row));
    // MUPIX DAC block 2 (10.5)
    for(const auto& para : parameters_chipdacs2)
        addPara(para, "");
    // MUPIX DAC block Voltage DACs
    for(const auto& para : parameters_voltagedacs)
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
    delete[] bitpattern_r;
    delete[] bitpattern_w;
}

void MupixConfig::Parse_ChipDACs_from_struct(MUPIX_CHIPDACS mt_g){
    setParameter("Bandgap1_on", mt_g.Bandgap1_on);
    setParameter("Biasblock1_on", mt_g.Biasblock1_on);
    setParameter("VNRegCasc", mt_g.VNRegCasc);
    setParameter("VDel", mt_g.VDel);
    setParameter("VPComp", mt_g.VPComp);
    setParameter("VPDAC", mt_g.VPDAC);
    setParameter("BLResDig", mt_g.BLResDig);
    setParameter("VPVCO", mt_g.VPVCO);
    setParameter("VNVCO", mt_g.VNVCO);
    setParameter("VPDelDclMux", mt_g.VPDelDclMux);
    setParameter("VNDelDclMux", mt_g.VNDelDclMux);
    setParameter("VPDelDcl", mt_g.VPDelDcl);
    setParameter("VNDelDcl", mt_g.VNDelDcl);
    setParameter("VPDelPreEmp", mt_g.VPDelPreEmp);
    setParameter("VNDelPreEmp", mt_g.VNDelPreEmp);
    setParameter("VPDcl", mt_g.VPDcl);
    setParameter("VNDcl", mt_g.VNDcl);
    setParameter("VNLVDS", mt_g.VNLVDS);
    setParameter("VNLVDSDel", mt_g.VNLVDSDel);
    setParameter("VPPump", mt_g.VPPump);
    setParameter("resetckdivend", mt_g.resetckdivend);
    setParameter("maxcycend", mt_g.maxcycend);
    setParameter("slowdownend", mt_g.slowdownend);
    setParameter("timerend", mt_g.timerend);
    setParameter("tsphase", mt_g.tsphase);
    setParameter("ckdivend2", mt_g.ckdivend2);
    setParameter("ckdivend", mt_g.ckdivend);
    setParameter("VPRegCasc", mt_g.VPRegCasc);
    setParameter("VPRamp", mt_g.VPRamp);
    setParameter("VPBiasReg", mt_g.VPBiasReg);
    setParameter("VNBiasReg", mt_g.VNBiasReg);
    setParameter("enable2threshold", mt_g.enable2threshold);
    setParameter("enableADC", mt_g.enableADC);
    setParameter("Invert", mt_g.Invert);
    setParameter("SelEx", mt_g.SelEx);
    setParameter("SelSlow", mt_g.SelSlow);
    setParameter("EnablePLL", mt_g.EnablePLL);
    setParameter("Readout_reset_n", mt_g.Readout_reset_n);
    setParameter("Serializer_reset_n", mt_g.Serializer_reset_n);
    setParameter("Aurora_reset_n", mt_g.Aurora_reset_n);
    setParameter("sendcounter", mt_g.sendcounter);
    setParameter("Linkselect", mt_g.Linkselect);
    setParameter("Termination", mt_g.Termination);
    setParameter("AlwaysEnable", mt_g.AlwaysEnable);
    setParameter("SelectTest", mt_g.SelectTest);
    setParameter("SelectTestOut", mt_g.SelectTestOut);
    setParameter("DisableHitbus", mt_g.DisableHitbus);
}

void MupixConfig::Parse_DigiRowDACs_from_struct(MUPIX_DIGIROWDACS& mt_ch, int channel){
    setParameter("digiWrite_" + std::to_string(channel), mt_ch.digiWrite);
}

void MupixConfig::Parse_RowDACs_from_struct(MUPIX_ROWDACS& mt_ch, int channel) {
    //setParameter("unused_" + std::to_string(channel), mt_ch.unused);
    setParameter("EnableInjection_" + std::to_string(channel), mt_ch.EnableInjection);
    setParameter("EnableAnalogueBuffer_" + std::to_string(channel), mt_ch.EnableAnalogueBuffer);
}

void MupixConfig::Parse_ColDACs_from_struct(MUPIX_COLDACS& mt_ch, int channel) {
    setParameter("RAM_" + std::to_string(channel), mt_ch.RAM);
    setParameter("EnableInjection_" + std::to_string(channel), mt_ch.EnableInjection);
    setParameter("EnableHitbus_" + std::to_string(channel), mt_ch.EnableHitbus);
}

void MupixConfig::Parse_ChipDACs2_from_struct(MUPIX_CHIPDACS2 mt_g){

    setParameter("Bandgap2_on", mt_g.Bandgap2_on);
    setParameter("Biasblock2_on", mt_g.Biasblock2_on);
    setParameter("BLResPix", mt_g.BLResPix);
    setParameter("VNPix", mt_g.VNPix);
    setParameter("VNFBPix", mt_g.VNFBPix);
    setParameter("VNFollPix", mt_g.VNFollPix);
    setParameter("VNPix2", mt_g.VNPix2);
    setParameter("VNBiasPix", mt_g.VNBiasPix);
    setParameter("VPLoadPix", mt_g.VPLoadPix);
    setParameter("VNOutPix", mt_g.VNOutPix);
    setParameter("VPFoll", mt_g.VPFoll);
    setParameter("VNDACPix", mt_g.VNDACPix);
}

void MupixConfig::Parse_VoltageDACs_from_struct(MUPIX_VOLTAGEDACS mt_g) {
    setParameter("ThLow", mt_g.ThLow);
    setParameter("ThPix", mt_g.ThPix);
    setParameter("BLPix", mt_g.BLPix);
    setParameter("BLDig", mt_g.BLDig);
    setParameter("ThHigh", mt_g.ThHigh);
}

/// MUPIX board configuration

MupixBoardConfig::paras_t MupixBoardConfig::parameters_boarddacs = {
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
    //setParameter("ext_trig_mode", mt_g.ext_trig_mode);
    //setParameter("ext_trig_endtime_sign", mt_g.ext_trig_endtime_sign);
}


} // namespace mutrig
