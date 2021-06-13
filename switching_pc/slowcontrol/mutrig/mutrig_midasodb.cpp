#include <cstring>
#include <iostream>
#include <iomanip>
#include <midas.h>
#include "odbxx.h"
#include <history.h>
//#include "experim.h"
#include "MutrigConfig.h"
#include "mutrig_MIDAS_config.h"
#include "mutrig_midasodb.h"

using midas::odb;

namespace mutrig { namespace midasODB {
//#ifdef DEBUG_VERBOSE
//#define ddprintf(args...) printf(args)
//#else
//#define ddprintf(args...)
//#endif


int setup_db(const char* prefix, MutrigFEB* FEB_interface){
    /* Book Setting space */
    INT status = DB_SUCCESS;

    char set_str[255];
    /* Add [prefix]/ASICs/Global (structure defined in mutrig_MIDAS_config.h) */
    //TODO some globals should be per asic
    sprintf(set_str, "%s/Settings/ASICs/Global", prefix);
    auto settings_asics = MUTRIG_GLOBAL_SETTINGS;
     // global mutrig setting are from mutrig_MIDAS_config.h
    settings_asics.connect(set_str);
 
    //Set number of ASICs, derived from mapping
    unsigned int nasics = FEB_interface->GetNumASICs();
    sprintf(set_str, "%s/Settings/Daq", prefix);
    if(nasics == 0){
        cm_msg(MINFO, "mutrig_midasodb::setup_db", "Number of ASICs is 0, will not continue to build DB. Consider to delete ODB subtree %s", prefix);
        return DB_SUCCESS;
    }
    cm_msg(MINFO, "mutrig_midasodb::setup_db", "For ODB subtree %s, number of ASICs is set to %u", prefix, nasics);

    // Add [prefix]/Daq (structure defined in mutrig_MIDAS_config.h) 
    auto settings_daq = MUTRIG_DAQ_SETTINGS; // gloabl setting for daq/fpga from mutrig_MIDAS_config.h
    settings_daq.connect(set_str);
    //update length flags for DAQ section
    settings_daq["num_asics"]=nasics;
    settings_daq["mask"].resize(nasics);
    settings_daq["resetskew_cphase"].resize(FEB_interface->GetNumModules());
    settings_daq["resetskew_cdelay"].resize(FEB_interface->GetNumModules());
    settings_daq["resetskew_phases"].resize(FEB_interface->GetNumModules());
    settings_daq.connect(set_str);


    // use lambda function for passing FEB_interface
    auto on_settings_changed_partial =
            [&FEB_interface](odb o) { 
                return MutrigFEB::on_settings_changed(
                    o, FEB_interface
                );
            };
    settings_daq.watch(on_settings_changed_partial);

    /* Map Equipment/SciFi/ASICs/TDCs and /Equipment/Scifi/ASICs/Channels
     * (structure defined in mutrig_MIDAS_config.h) */
    auto settings_tdc = MUTRIG_TDC_SETTINGS;
    auto settings_ch = MUTRIG_CH_SETTINGS;
    for(unsigned int asic = 0; asic < nasics; ++asic) {
        sprintf(set_str, "%s/Settings/ASICs/TDCs/%i", prefix, asic);
        settings_tdc.connect(set_str);
        for(unsigned int ch = 0; ch < 32; ++ch) {
            sprintf(set_str, "%s/Settings/ASICs/Channels/%i", prefix, asic*32+ch);
            settings_ch.connect(set_str);
        }
    }

    //set up variables read from FEB: counters
    sprintf(set_str, "%s/Variables/Counters", prefix);
    odb variables_counters = {
        {"nHits", std::array<uint32_t, 255>()},
        {"Time", std::array<uint32_t, 255>()},
        {"nBadFrames", std::array<uint32_t, 255>()},
        {"nFrames", std::array<uint32_t, 255>()},
        {"nErrorsLVDS", std::array<uint32_t, 255>()},
        {"nWordsLVDS", std::array<uint32_t, 255>()},
        {"nErrorsPRBS", std::array<uint32_t, 255>()},
        {"nWordsPRBS", std::array<uint32_t, 255>()},
        {"nDatasyncloss", std::array<uint32_t, 255>()},
    };
    variables_counters.connect(set_str);
    variables_counters["nHits"].resize(nasics);
    variables_counters["Time"].resize(nasics);
    variables_counters["nBadFrames"].resize(nasics);
    variables_counters["nFrames"].resize(nasics);
    variables_counters["nErrorsLVDS"].resize(nasics);
    variables_counters["nWordsLVDS"].resize(nasics);
    variables_counters["nErrorsPRBS"].resize(nasics);
    variables_counters["nWordsPRBS"].resize(nasics);
    variables_counters["nDatasyncloss"].resize(nasics);

    variables_counters.connect(set_str);

    //set up variables read from FEB: run state & reset system bypass
    sprintf(set_str, "%s/Variables/FEB Run State", prefix);
    odb bypass_setting = {
            {"Bypass enabled", std::array<uint32_t, 255>()},
            {"Run state", std::array<uint32_t, 255>()}
    };
    bypass_setting.connect(set_str);
    bypass_setting["Bypass enabled"].resize(FEB_interface->GetNumFPGAs());
    bypass_setting["Run state"].resize(FEB_interface->GetNumFPGAs());
    bypass_setting.connect(set_str);

    //set up variables read from FEB: run state & reset system bypass
    sprintf(set_str, "%s/Variables/FEB datapath status", prefix);
    odb datapath_status = {
            {"PLL locked", std::array<uint32_t, 255>()},
            {"Buffer full", std::array<uint32_t, 255>()},
            {"Frame desync", std::array<uint32_t, 255>()},
            {"DPA locked", std::array<uint32_t, 255>()},
            {"RX ready", std::array<uint32_t, 255>()}
    };
    datapath_status.connect(set_str);
    datapath_status["PLL locked"].resize(FEB_interface->GetNumFPGAs());
    datapath_status["Buffer full"].resize(FEB_interface->GetNumFPGAs());
    datapath_status["Frame desync"].resize(FEB_interface->GetNumFPGAs());
    datapath_status["DPA locked"].resize(FEB_interface->GetNumASICs());
    datapath_status["RX ready"].resize(FEB_interface->GetNumASICs());
    datapath_status.connect(set_str);

    // Define history panels
    for(std::string panel: {
       "Counters_nHits",
       "Counters_nFrames",
       "Counters_nWordsLVDS",
       "Counters_nWordsPRBS",
       "Counters_nBadFrames",
       "Counters_nErrorsLVDS",
       "Counters_nErrorsPRBS",
       "Counters_nDatasyncloss",
       "FEB datapath status_RX ready"
       }){
        std::vector<std::string> varlist;
        char name[255];
        for(int i = 0; i < FEB_interface->GetNumASICs(); i++){
            snprintf(name,255,"%s:%s[%d]",FEB_interface->GetName(),panel.c_str(),i);
            varlist.push_back(name);
        }
        hs_define_panel(FEB_interface->GetName(),panel.c_str(),varlist);
    }

    //hs_define_panel("SciFi","Times",{"SciFi:Counters_Time",
    //                                "SciFi:Counters_Time"});

    return status;
}

mutrig::MutrigConfig MapConfigFromDB(const char* prefix, int asic) {

    MutrigConfig ret;
    ret.reset();

    char set_str[255];

    sprintf(set_str, "%s/Settings/ASICs", prefix);
    //TODO: Can we avoid this silly back and forth casting?
    odb settings_asics(std::string(set_str).c_str());

    // get global asic settings from odb;
    ret.Parse_GLOBAL_from_struct(settings_asics["Global"]);

    // get tdcs asic settings from odb
    sprintf(set_str, "TDCs/%i", asic);
    ret.Parse_TDC_from_struct(settings_asics[set_str]);

    // get channels asic settings from odb
    for(int ch = 0; ch < 32 ; ch++) {
        sprintf(set_str, "Channels/%i", asic*32+ch);
        ret.Parse_CH_from_struct(settings_asics[set_str], ch);
    }

    return ret;
}

} } // namespace mutrig::midasODB
