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


int setup_db(std::string prefix, MutrigFEB & FEB_interface, uint32_t nasics, uint32_t nModules, uint32_t nAsicsPerFeb, bool write_defaults = true){
    /* Book Setting space */
    INT status = DB_SUCCESS;

    /* Add [prefix]/ASICs/Global (structure defined in mutrig_MIDAS_config.h) */
    //TODO some globals should be per asic
    auto settings_asics = MUTRIG_GLOBAL_SETTINGS;
     // global mutrig setting are from mutrig_MIDAS_config.h
    settings_asics.connect(prefix + "/Settings/ASICs/Global");

    if(nasics == 0){
        cm_msg(MINFO, "mutrig_midasodb::setup_db", "Number of MuTRiGs is 0, will not continue to build DB. Consider to delete ODB subtree %s", prefix.c_str());
        return DB_SUCCESS;
    }
    cm_msg(MINFO, "mutrig_midasodb::setup_db", "For ODB subtree %s, number of ASICs is set to %u", prefix.c_str(), nasics);

    // Add [prefix]/Daq (structure defined in mutrig_MIDAS_config.h) 
    auto settings_daq = MUTRIG_DAQ_SETTINGS; // gloabl setting for daq/fpga from mutrig_MIDAS_config.h
    settings_daq.connect(prefix + "/Settings/Daq", write_defaults);
    //update length flags for DAQ section
    settings_daq["num_asics"]=nasics;
    settings_daq["num_modules_per_feb"]=nModules;
    settings_daq["num_asics_per_module"]=nAsicsPerFeb;
    settings_daq["mask"].resize(nasics);
    settings_daq["resetskew_cphase"].resize(nModules);
    settings_daq["resetskew_cdelay"].resize(nModules);
    settings_daq["resetskew_phases"].resize(nModules);
    settings_daq.connect(prefix + "/Settings/Daq", write_defaults);

    auto commands = ScifiCentralCommands;
    commands.connect(prefix + "/Commands", write_defaults);

    // use lambda function for passing FEB_interface
    // TODO: don't set watch here for the moment use the one in switch_fe
//    auto on_settings_changed_partial =
//            [&FEB_interface](odb o) {
//                return MutrigFEB::on_settings_changed(
//                    o, FEB_interface
//                );
//            };

//    settings_daq.watch(on_settings_changed_partial);

    /* Map Equipment/SciFi/ASICs/TDCs and /Equipment/Scifi/ASICs/Channels
     * (structure defined in mutrig_MIDAS_config.h) */
    auto settings_tdc = MUTRIG_TDC_SETTINGS;
    auto settings_ch = MUTRIG_CH_SETTINGS;
    for(unsigned int asic = 0; asic < nasics; ++asic) {
        settings_tdc.connect(prefix+"/Settings/ASICs/TDCs/"+std::to_string(asic));
        // TODO: Get rid of explicit 32 here, maybe oranize ODB miore nicely
        for(unsigned int ch = 0; ch < 32; ++ch) {
            settings_ch.connect(prefix+"/Settings/ASICs/Channels/"+std::to_string(asic*32+ch));
        }
    }

    //set up variables read from FEB: counters
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
        {"Rate", std::array<uint32_t,128>()}, //add rate
    };
    variables_counters.connect(prefix + "/Variables/Counters");
    variables_counters["nHits"].resize(nasics);
    variables_counters["Time"].resize(nasics);
    variables_counters["nBadFrames"].resize(nasics);
    variables_counters["nFrames"].resize(nasics);
    variables_counters["nErrorsLVDS"].resize(nasics);
    variables_counters["nWordsLVDS"].resize(nasics);
    variables_counters["nErrorsPRBS"].resize(nasics);
    variables_counters["nWordsPRBS"].resize(nasics);
    variables_counters["nDatasyncloss"].resize(nasics);
    variables_counters["Rate"].resize(nasics); //add rate

    variables_counters.connect(prefix + "/Variables/Counters");

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
        for(int i = 0; i < FEB_interface.GetNumASICs(); i++){
            snprintf(name,255,"%s:%s[%d]",FEB_interface.GetName().c_str(),panel.c_str(),i);
            varlist.push_back(name);
        }
        hs_define_panel(FEB_interface.GetName().c_str(),panel.c_str(),varlist);
    }

    return status;
}

mutrig::MutrigConfig MapConfigFromDB(odb settings_asics, int asic) {

    MutrigConfig ret;
    ret.reset();

    // get global asic settings from odb;
    ret.Parse_GLOBAL_from_struct(settings_asics["Global"]);

    // get tdcs asic settings from odb
    std::string path ="TDCs/"+std::to_string(asic);
    ret.Parse_TDC_from_struct(settings_asics[path]);

    // get channels asic settings from odb
    // TODO: Get rid of the explicit 32
    for(int ch = 0; ch < 32 ; ch++) {
        std::string path2 ="Channels/" + std::to_string(asic*32+ch);
        ret.Parse_CH_from_struct(settings_asics[path2], ch);
    }

    return ret;
}

} } // namespace mutrig::midasODB
