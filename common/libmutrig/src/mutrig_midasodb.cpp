#include <cstring>
#include <iostream>
#include <iomanip>
#include <midas.h>
#include <history.h>
//#include "experim.h"
#include "MutrigConfig.h"
#include "mutrig_MIDAS_config.h"
#include "mutrig_midasodb.h"
namespace mutrig { namespace midasODB {
//#ifdef DEBUG_VERBOSE
//#define ddprintf(args...) printf(args)
//#else
//#define ddprintf(args...)
//#endif


int setup_db(HNDLE& hDB, const char* prefix, MutrigFEB* FEB_interface){
    /* Book Setting space */
printf("setting up db\n");
    HNDLE hTmp;
    INT status = DB_SUCCESS;
    char set_str[255];

    /* Add [prefix]/ASICs/Global (structure defined in mutrig_MIDAS_config.h) */
    //TODO some globals should be per asic
    MUTRIG_GLOBAL_STR(mutrig_global);   // global mutrig settings
    sprintf(set_str, "%s/Settings/ASICs/Global", prefix);
    //ddprintf("mutrig_midasodb: adding struct %s\n",set_str);
    status = db_create_record(hDB, 0, set_str, strcomb(mutrig_global));
    status = db_find_key (hDB, 0, set_str, &hTmp);
    if (status != DB_SUCCESS) {
        cm_msg(MINFO,"mutrig_midasodb", "Key %s not found", set_str);
        return status;
    }

    //Set number of ASICs, derived from mapping
    unsigned int nasics=FEB_interface->GetNumASICs();
    sprintf(set_str, "%s/Settings/Daq/num_asics", prefix);
    INT ival=nasics;
    if((status = db_set_value(hDB ,0,set_str, &ival, sizeof(INT), 1, TID_INT))!=DB_SUCCESS) return status;

    if(nasics==0){
        cm_msg(MINFO,"mutrig_midasodb::setup_db","Number of ASICs is 0, will not continue to build DB. Consider to delete ODB subtree %s",prefix);
    return DB_SUCCESS;
    }

    /* Add [prefix]/Daq (structure defined in mutrig_MIDAS_config.h) */
    //TODO: if we have more than one FE-FPGA, there might be more than one DAQ class.
    MUTRIG_DAQ_STR(mutrig_daq);         // global settings for daq/fpga
    sprintf(set_str, "%s/Settings/Daq", prefix);
    status = db_create_record(hDB, 0, set_str, strcomb(mutrig_daq));
    status = db_find_key (hDB, 0, set_str, &hTmp);
    if (status != DB_SUCCESS) {
        cm_msg(MINFO,"mutrig_midasodb", "Key %s not found", set_str);
        return status;
    }
    //open hot link
    db_watch(hDB, hTmp, &MutrigFEB::on_settings_changed, FEB_interface);

    //update length flags for DAQ section
    sprintf(set_str, "%s/Settings/Daq/mask", prefix);
    if((status = db_find_key (hDB, 0, set_str, &hTmp))!=DB_SUCCESS) return status;
    if((status = db_set_num_values(hDB, hTmp, nasics))!=DB_SUCCESS) return status;

    sprintf(set_str, "%s/Settings/Daq/resetskew_cphase", prefix);
    if((status = db_find_key (hDB, 0, set_str, &hTmp))!=DB_SUCCESS) return status;
    if((status = db_set_num_values(hDB, hTmp, FEB_interface->GetNumModules()))!=DB_SUCCESS) return status;

    sprintf(set_str, "%s/Settings/Daq/resetskew_cdelay", prefix);
    if((status = db_find_key (hDB, 0, set_str, &hTmp))!=DB_SUCCESS) return status;
    if((status = db_set_num_values(hDB, hTmp, FEB_interface->GetNumModules()))!=DB_SUCCESS) return status;

    sprintf(set_str, "%s/Settings/Daq/resetskew_phases", prefix);
    if((status = db_find_key (hDB, 0, set_str, &hTmp))!=DB_SUCCESS) return status;
    if((status = db_set_num_values(hDB, hTmp, FEB_interface->GetNumModules()))!=DB_SUCCESS) return status;


    /* Map Equipment/SciFi/ASICs/TDCs and /Equipment/Scifi/ASICs/Channels
     * (structure defined in mutrig_MIDAS_config.h) */
    MUTRIG_TDC_STR(mutrig_tdc_str);
    MUTRIG_CH_STR(mutrig_ch_str);
    for(unsigned int asic = 0; asic < nasics; ++asic) {
        sprintf(set_str, "%s/Settings/ASICs/TDCs/%i", prefix, asic);
        //ddprintf("mutrig_midasodb: adding struct %s\n",set_str);
        status = db_create_record(hDB, 0, set_str, strcomb(mutrig_tdc_str));
        status = db_find_key (hDB, 0, set_str, &hTmp);
        if (status != DB_SUCCESS) {
            cm_msg(MINFO,"frontend_init", "Key %s not found", set_str);
            return status;
        }

        for(unsigned int ch = 0; ch < 32; ++ch) {
            sprintf(set_str, "%s/Settings/ASICs/Channels/%i",
                    prefix, asic*32+ch);
            //ddprintf("mutrig_midasodb: adding struct %s\n",set_str);
            status = db_create_record(hDB, 0, set_str, strcomb(mutrig_ch_str));
            status = db_find_key (hDB, 0, set_str, &hTmp);
            if (status != DB_SUCCESS) {
                cm_msg(MINFO,"frontend_init", "Key %s not found", set_str);
                return status;
            }
        }
    }

    //set up variables read from FEB: counters
    sprintf(set_str, "%s/Variables/Counters/nHits", prefix);
    status=db_create_key(hDB, 0, set_str, TID_DWORD);
    if (!(status==DB_SUCCESS || status==DB_KEY_EXIST)) return status;
    if((status = db_find_key (hDB, 0, set_str, &hTmp))!=DB_SUCCESS) return status;
    if((status = db_set_num_values(hDB, hTmp, nasics))!=DB_SUCCESS) return status;

    sprintf(set_str, "%s/Variables/Counters/Time", prefix);
    status=db_create_key(hDB, 0, set_str, TID_DWORD);
    if (!(status==DB_SUCCESS || status==DB_KEY_EXIST)) return status;
    if((status = db_find_key (hDB, 0, set_str, &hTmp))!=DB_SUCCESS) return status;
    if((status = db_set_num_values(hDB, hTmp, nasics))!=DB_SUCCESS) return status;

    sprintf(set_str, "%s/Variables/Counters/nBadFrames", prefix);
    status=db_create_key(hDB, 0, set_str, TID_DWORD);
    if (!(status==DB_SUCCESS || status==DB_KEY_EXIST)) return status;
    if((status = db_find_key (hDB, 0, set_str, &hTmp))!=DB_SUCCESS) return status;
    if((status = db_set_num_values(hDB, hTmp, nasics))!=DB_SUCCESS) return status;

    sprintf(set_str, "%s/Variables/Counters/nFrames", prefix);
    status=db_create_key(hDB, 0, set_str, TID_DWORD);
    if (!(status==DB_SUCCESS || status==DB_KEY_EXIST)) return status;
    if((status = db_find_key (hDB, 0, set_str, &hTmp))!=DB_SUCCESS) return status;
    if((status = db_set_num_values(hDB, hTmp, nasics))!=DB_SUCCESS) return status;

    sprintf(set_str, "%s/Variables/Counters/nErrorsLVDS", prefix);
    status=db_create_key(hDB, 0, set_str, TID_DWORD);
    if (!(status==DB_SUCCESS || status==DB_KEY_EXIST)) return status;
    if((status = db_find_key (hDB, 0, set_str, &hTmp))!=DB_SUCCESS) return status;
    if((status = db_set_num_values(hDB, hTmp, nasics))!=DB_SUCCESS) return status;

    sprintf(set_str, "%s/Variables/Counters/nWordsLVDS", prefix);
    status=db_create_key(hDB, 0, set_str, TID_DWORD);
    if (!(status==DB_SUCCESS || status==DB_KEY_EXIST)) return status;
    if((status = db_find_key (hDB, 0, set_str, &hTmp))!=DB_SUCCESS) return status;
    if((status = db_set_num_values(hDB, hTmp, nasics))!=DB_SUCCESS) return status;

    sprintf(set_str, "%s/Variables/Counters/nErrorsPRBS", prefix);
    status=db_create_key(hDB, 0, set_str, TID_DWORD);
    if (!(status==DB_SUCCESS || status==DB_KEY_EXIST)) return status;
    if((status = db_find_key (hDB, 0, set_str, &hTmp))!=DB_SUCCESS) return status;
    if((status = db_set_num_values(hDB, hTmp, nasics))!=DB_SUCCESS) return status;

    sprintf(set_str, "%s/Variables/Counters/nWordsPRBS", prefix);
    status=db_create_key(hDB, 0, set_str, TID_DWORD);
    if (!(status==DB_SUCCESS || status==DB_KEY_EXIST)) return status;
    if((status = db_find_key (hDB, 0, set_str, &hTmp))!=DB_SUCCESS) return status;
    if((status = db_set_num_values(hDB, hTmp, nasics))!=DB_SUCCESS) return status;

    sprintf(set_str, "%s/Variables/Counters/nDatasyncloss", prefix);
    status=db_create_key(hDB, 0, set_str, TID_DWORD);
    if (!(status==DB_SUCCESS || status==DB_KEY_EXIST)) return status;
    if((status = db_find_key (hDB, 0, set_str, &hTmp))!=DB_SUCCESS) return status;
    if((status = db_set_num_values(hDB, hTmp, nasics))!=DB_SUCCESS) return status;


    //set up variables read from FEB: run state & reset system bypass
    const char *bypass_settings_str[] = {
    "Bypass enabled = BOOL[2] :",\
    "[0] n",\
    "[1] n",\
    "Run state = DWORD[2] :",\
    "[0] 255",\
    "[1] 255",\
    "",\
    NULL};

    sprintf(set_str, "%s/Variables/FEB Run State", prefix);
    db_create_record(hDB, 0, set_str, strcomb(bypass_settings_str));

    sprintf(set_str, "%s/Variables/FEB Run State/Bypass enabled", prefix);
    db_find_key (hDB, 0, set_str, &hTmp);
    assert(hTmp);
    if((status = db_set_num_values(hDB, hTmp, FEB_interface->GetNumFPGAs()))!=DB_SUCCESS) return status;

    sprintf(set_str, "%s/Variables/FEB Run State/Run state", prefix);
    db_find_key (hDB, 0, set_str, &hTmp);
    assert(hTmp);
    if((status = db_set_num_values(hDB, hTmp, FEB_interface->GetNumFPGAs()))!=DB_SUCCESS) return status;
    DWORD val=0xff;
    for(int i=0;i<FEB_interface->GetNumFPGAs();i++)
    	if((status = db_set_value_index(hDB,0,set_str, &val, sizeof(DWORD),i, TID_DWORD,false))!=DB_SUCCESS) return status;

    //set up variables read from FEB: run state & reset system bypass
    const char *datapath_status_str[] = {
    "PLL locked = BOOL[2] :",\
    "[0] n",\
    "[1] n",\
    "Buffer full = BOOL[2] :",\
    "[0] n",\
    "[1] n",\
    "Frame desync = BOOL[2] :",\
    "[0] n",\
    "[1] n",\
    "DPA locked = BOOL[2] :",\
    "[0] n",\
    "[1] n",\
    "RX ready = BOOL[2] :",\
    "[0] n",\
    "[1] n",\
    "",\
    NULL};

    sprintf(set_str, "%s/Variables/FEB datapath status", prefix);
    db_create_record(hDB, 0, set_str, strcomb(datapath_status_str));

    sprintf(set_str, "%s/Variables/FEB datapath status/PLL locked", prefix);
    db_find_key (hDB, 0, set_str, &hTmp);
    if((status = db_set_num_values(hDB, hTmp, FEB_interface->GetNumFPGAs()))!=DB_SUCCESS) return status;

    sprintf(set_str, "%s/Variables/FEB datapath status/Buffer full", prefix);
    db_find_key (hDB, 0, set_str, &hTmp);
    if((status = db_set_num_values(hDB, hTmp, FEB_interface->GetNumFPGAs()))!=DB_SUCCESS) return status;

    sprintf(set_str, "%s/Variables/FEB datapath status/Frame desync", prefix);
    db_find_key (hDB, 0, set_str, &hTmp);
    if((status = db_set_num_values(hDB, hTmp, FEB_interface->GetNumFPGAs()))!=DB_SUCCESS) return status;

    sprintf(set_str, "%s/Variables/FEB datapath status/DPA locked", prefix);
    db_find_key (hDB, 0, set_str, &hTmp);
    if((status = db_set_num_values(hDB, hTmp, FEB_interface->GetNumASICs()))!=DB_SUCCESS) return status;

    sprintf(set_str, "%s/Variables/FEB datapath status/DPA locked", prefix);
    db_find_key (hDB, 0, set_str, &hTmp);
    if((status = db_set_num_values(hDB, hTmp, FEB_interface->GetNumASICs()))!=DB_SUCCESS) return status;

    sprintf(set_str, "%s/Variables/FEB datapath status/RX ready", prefix);
    db_find_key (hDB, 0, set_str, &hTmp);
    if((status = db_set_num_values(hDB, hTmp, FEB_interface->GetNumASICs()))!=DB_SUCCESS) return status;


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
        for(int i=0;i<FEB_interface->GetNumASICs();i++){
            snprintf(name,255,"%s:%s[%d]",FEB_interface->GetName(),panel.c_str(),i);
            varlist.push_back(name);
        }
        hs_define_panel(FEB_interface->GetName(),panel.c_str(),varlist);
    }

    //hs_define_panel("SciFi","Times",{"SciFi:Counters_Time",
    //                                "SciFi:Counters_Time"});

return status;
}

mutrig::MutrigConfig MapConfigFromDB(HNDLE& db_rootentry, const char* prefix, int asic) {
    MutrigConfig ret;
    ret.reset();
    char set_str[255];
    int status, size;
    // structs from ODB

    HNDLE hGlob;
    MUTRIG_GLOBAL mt_global;
    sprintf(set_str, "%s/Settings/ASICs/Global", prefix);
    //ddprintf("mutrig_midasodb: Mapping ODB to Config for %s, asic %d: Using key %s as global\n",prefix,asic, set_str);
    // why not take key from setup_db?
    status = db_find_key(db_rootentry, 0, set_str, &hGlob);
    if(status != DB_SUCCESS) {
        cm_msg(MERROR, "mutrig::midasODB::MapConfigFromDB", "Cannot find key %s", set_str);
    }

    size = sizeof(mt_global);
    status = db_get_record(db_rootentry, hGlob, &mt_global, &size, 0);
    if(status != DB_SUCCESS) {
        cm_msg(MERROR, "mutrig::midasODB::MapConfigFromDB", "Cannot retrieve MuTRiG global settings");
    }
    ret.Parse_GLOBAL_from_struct(mt_global);

    HNDLE hTDC;
    MUTRIG_TDC mt_tdc;
    sprintf(set_str, "%s/Settings/ASICs/TDCs/%i", prefix, asic);
    //ddprintf("mutrig_midasodb: Mapping ODB to Config for %s, asic %d: Using key %s as TDC\n",prefix,asic, set_str);
    status = db_find_key(db_rootentry, 0, set_str, &hTDC);
    if(status != DB_SUCCESS) {
        cm_msg(MERROR, "mutrig::midasODB::MapConfigFromDB", "Cannot find key %s", set_str);
    }
    size = sizeof(mt_tdc);
    status = db_get_record(db_rootentry, hTDC, &mt_tdc, &size, 0);
    if(status != DB_SUCCESS) {
        cm_msg(MERROR, "mutrig::midasODB::MapConfigFromDB", "Cannot retrieve MuTRiG tdc settings");
    }
    ret.Parse_TDC_from_struct(mt_tdc);

    HNDLE hCh;
    MUTRIG_CH mt_ch;
    for(int ch = 0; ch < 32 ; ch++) {
        sprintf(set_str, "%s/Settings/ASICs/Channels/%i", prefix, asic*32+ch);
        //ddprintf("mutrig_midasodb: Mapping ODB to Config for %s, asic %d: Using key %s as channel %d\n",prefix,asic, set_str,ch);
        status = db_find_key(db_rootentry, 0, set_str, &hCh);
        if(status != DB_SUCCESS) {
            cm_msg(MERROR, "mutrig::midasODB::MapConfigFromDB", "Cannot find key %s", set_str);
        }
        size = sizeof(mt_ch);
        status = db_get_record(db_rootentry, hCh, &mt_ch, &size, 0);
        if(status != DB_SUCCESS) {
            cm_msg(MERROR, "mutrig::midasODB::MapConfigFromDB", "Cannot retrieve MuTRiG ch settings");
        }
        ret.Parse_CH_from_struct(mt_ch, ch);
    }


    return ret;
}

} } // namespace mutrig::midasODB
