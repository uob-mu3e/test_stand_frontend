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

    cm_msg(MINFO, "mutrig_midasobb::setup_db", "Setting up odb");
    INT status = DB_SUCCESS;

    char set_str[255];

    /* Add [prefix]/ASICs/Global (structure defined in mutrig_MIDAS_config.h) */
    //TODO some globals should be per asic
    sprintf(set_str, "%s/Settings/ASICs/Global", prefix);
    auto settings_asics = MUTRIG_GLOBAL_SETTINGS;
    settings_asics.connect(set_str, true); // global mutrig setting are from mutrig_MIDAS_config.h
 
    //Set number of ASICs, derived from mapping
    unsigned int nasics = FEB_interface->GetNumASICs();
    sprintf(set_str, "%s/Settings/Daq", prefix);
    odb settings_daq_nasics0 = {{"num_asics", nasics}}; // 
    settings_daq_nasics0.connect(set_str, true);
    if(nasics == 0){
        cm_msg(MINFO, "mutrig_midasodb::setup_db", "Number of ASICs is 0, will not continue to build DB. Consider to delete ODB subtree %s", prefix);
        return DB_SUCCESS;
    }

    // Add [prefix]/Daq (structure defined in mutrig_MIDAS_config.h) 
    //TODO: if we have more than one FE-FPGA, there might be more than one DAQ class.
    auto settings_daq = MUTRIG_DAQ_SETTINGS; // gloabl setting for daq/fpga from mutrig_MIDAS_config.h
    settings_daq.connect(set_str, true);

    // use lambda function for passing FEB_interface
    auto on_settings_changed_partial =
            [&FEB_interface](odb o) { return MutrigFEB::on_settings_changed(o, FEB_interface); };
    settings_daq.watch(on_settings_changed_partial);

    //update length flags for DAQ section
    odb settings_daq_nasicsn = {
        {"mask", nasics},
        {"resetskew_cphase", FEB_interface->GetNumModules()},
        {"resetskew_cdelay", FEB_interface->GetNumModules()},
        {"resetskew_phases", FEB_interface->GetNumModules()},
    };
    settings_daq_nasicsn.connect(set_str);

    /* Map Equipment/SciFi/ASICs/TDCs and /Equipment/Scifi/ASICs/Channels
     * (structure defined in mutrig_MIDAS_config.h) */
    auto settings_tdc = MUTRIG_TDC_SETTINGS;
    auto settings_ch = MUTRIG_CH_SETTINGS;
    for(unsigned int asic = 0; asic < nasics; ++asic) {
        sprintf(set_str, "%s/Settings/ASICs/Global/TDCs/%i", prefix, asic);
        settings_tdc.connect(set_str, true);
        for(unsigned int ch = 0; ch < 32; ++ch) {
            sprintf(set_str, "%s/Settings/ASICs/Global/Channels/%i", prefix, asic*32+ch);
            settings_ch.connect(set_str, true);
        }
    }

    //set up variables read from FEB: counters
    sprintf(set_str, "%s/Variables/Counters", prefix);
    odb variables_counters = {
        {"nHits", nasics},
        {"Time", nasics},
        {"nBadFrames", nasics},
        {"nFrames", nasics},
        {"nErrorsLVDS", nasics},
        {"nWordsLVDS", nasics},
        {"nErrorsPRBS", nasics},
        {"nWordsPRBS", nasics},
        {"nDatasyncloss", nasics},
    };

    //set up variables read from FEB: run state & reset system bypass
    sprintf(set_str, "%s/Variables/FEB Run State", prefix);
    odb bypass_setting = {
            {"Bypass enabled", FEB_interface->GetNumFPGAs()},
            {"Run state", FEB_interface->GetNumFPGAs()}
    };
    bypass_setting.connect(set_str);

    //set up variables read from FEB: run state & reset system bypass
    sprintf(set_str, "%s/Variables/FEB datapath status", prefix);
    odb datapath_status = {
            {"PLL locked", FEB_interface->GetNumFPGAs()},
            {"Buffer full", FEB_interface->GetNumFPGAs()},
            {"Frame desync", FEB_interface->GetNumFPGAs()},
            {"DPA locked", FEB_interface->GetNumASICs()},
            {"RX ready", FEB_interface->GetNumASICs()}
    };
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
    odb settings_asics(set_str);

    // get global asic settings from odb;
    ret.Parse_GLOBAL_from_struct(settings_asics["Global&"]);

    // get tdcs asic settings from odb
    sprintf(set_str, "TDCs/%i&", asic);
    ret.Parse_TDC_from_struct(settings_asics[set_str]);

    // get channels asic settings from odb
    for(int ch = 0; ch < 32 ; ch++) {
        sprintf(set_str, "Channels/%i&", asic*32+ch);
        ret.Parse_CH_from_struct(settings_asics[set_str], ch);
    }

    return ret;
}

} } // namespace mutrig::midasODB
