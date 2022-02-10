#include <cstring>
#include <iostream>
#include <iomanip>

#include "midas.h"
#include "mupix_MIDAS_config.h"
#include "mupix_midasodb.h"
#include "odbxx.h"
#include "link_constants.h"
using midas::odb;
using namespace std;

namespace mupix { namespace midasODB {


int setup_db(std::string prefix, MupixFEB & FEB_interface, bool init_FEB, bool write_defaults = true){
    /* Book Setting space */
    
    cm_msg(MINFO, "mupix_midasodb::setup_db", "Setting up odb");
    
    /* Add [prefix]/ASICs/Global (structure defined in mutrig_MIDAS_config.h) */
    //TODO some globals should be per asic
    auto settings_asics_global = MUPIX_GLOBAL_SETTINGS;
    // global mupix settings from mupix_MIDAS_config.h
    settings_asics_global.connect(prefix + "/Settings/ASICs/Global", write_defaults);

    // set global FEB values
    auto global_settings_febs = MUPIX_GLOBAL_FEBS_SETTINGS;

    global_settings_febs.connect(prefix + "/Settings/FEBS", write_defaults);

    //Set number of ASICs, derived from mapping
    unsigned int nasics = FEB_interface.GetNumASICs();
    unsigned int nFEBs = FEB_interface.GetNumFPGAs();
    settings_asics_global["Num asics"] = nasics;
    settings_asics_global["Num boards"] = nFEBs;

    if(nasics==0){
        cm_msg(MINFO,"mupix_midasodb::setup_db","Number of Mupixes is 0, will not continue to build DB. Consider to delete ODB subtree %s",prefix.c_str());
        return DB_SUCCESS;
    }

    /* Add [prefix]/Daq (structure defined in mupix_MIDAS_config.h) */
    //TODO: if we have more than one FE-FPGA, there might be more than one DAQ class.
    auto settings_daq = MUPIX_DAQ_SETTINGS;
    settings_daq.connect(prefix + "/Settings/Daq", write_defaults);
    
    // use lambda funciton for passing FEB_interface
    auto on_settings_changed_partial = 
        [&FEB_interface](odb o) { 
            return MupixFEB::on_settings_changed(
                o, &FEB_interface
            );
        };
    settings_daq.watch(on_settings_changed_partial);

    //init all values on FEB
    if(init_FEB){
        //BOOL bval;

        //bval = settings_daq["dummy_config"];
        // TODO: do something here
        //FEB_interface->setDummyConfig(SciFiFEB::FPGA_broadcast_ID,bval); 
        
        //bval = settings_daq["dummy_data"];
        // TODO: do something here
        //FEB_interface->setDummyData_Enable(SciFiFEB::FPGA_broadcast_ID,bval);
        
        // TODO: do something here
        // for(int i=0;i<16;i++)
        // FEB_interface->setMask(i,settings_daq["mask"]i]);
    }

    // set all dac values per asic
    auto settings_biasdacs = MUPIX_BIASDACS_SETTINGS;
    auto settings_confdacs = MUPIX_CONFDACS_SETTINGS;
    auto settings_vdacs = MUPIX_VDACS_SETTINGS;
    auto settings_tdacs = MUPIX_TDACS_SETTINGS;

    nasics = settings_asics_global["Num asics"];
        
    for(unsigned int i = 0; i < nasics; ++i) {
        settings_biasdacs.connect(prefix +  "/Settings/BIASDACS/" + to_string(i), write_defaults);

        settings_confdacs.connect(prefix +  "/Settings/CONFDACS/" + to_string(i), write_defaults);

        settings_vdacs.connect(prefix +  "/Settings/VDACS/" + to_string(i), write_defaults);

        settings_tdacs.connect(prefix +  "/Settings/TDACS/" + to_string(i), write_defaults);
    }

    // set all tdac values per FEB
    auto settings_febs = MUPIX_FEB_SETTINGS;

    nFEBs = settings_asics_global["Num boards"];

    for(unsigned int i = 0; i < nFEBs; ++i) {
        settings_febs.connect(prefix +  "/Settings/FEBS/" + to_string(i), write_defaults);
    }

    // load tdac json from ODB into feb_interface
    FEB_interface.SetTDACs();

    // PSLL Bank setup
    /* Default values for /Equipment/Mupix/Settings */
    odb settings = {
        {namestrPSLL.c_str(), std::array<std::string, per_fe_PSLL_size*N_FEBS_MUPIX_INT_2021*lvds_links_per_feb>{}}
    };

    // TODO: why do I have to connect here? In switch_fe.cpp we do first the naming and than we connect
    settings.connect(prefix + "Settings", write_defaults);

    create_psll_names_in_odb(settings, N_FEBS_MUPIX_INT_2021, lvds_links_per_feb);

    settings.connect(prefix + "Settings", write_defaults=write_defaults);

    /* Default values for /Equipment/Mupix/Variables */
    odb variables = {
        {banknamePSLL.c_str(), std::array<int, per_fe_PSLL_size*N_FEBS_MUPIX_INT_2021*lvds_links_per_feb>{}}
    };

    variables.connect(prefix + "Variables", write_defaults=write_defaults);

    return DB_SUCCESS;
}


void create_psll_names_in_odb(odb & settings, uint32_t N_FEBS_MUPIX, uint32_t N_LINKS){
    int bankindex = 0;

    for(uint32_t i=0; i < N_FEBS_MUPIX; i++){
        for(uint32_t j=0; j < N_LINKS; j++){
            string name = "FEB" + to_string(i) + " LVDS" + to_string(j);

            string * s = new string(name);
            (*s) += " INDEX";
            settings[namestrPSLL.c_str()][bankindex++] = s;

            s = new string(name);
            (*s) += " STATUS LVDS";
            settings[namestrPSLL.c_str()][bankindex++] = s;

            s = new string(name);
            (*s) += " NUM HITS LVDS";
            settings[namestrPSLL.c_str()][bankindex++] = s;

            s = new string(name);
            (*s) += " NUM MuPix HITS LVDS";
            settings[namestrPSLL.c_str()][bankindex++] = s;
        }
    }
}


int MapForEachASIC(std::string prefix, std::function<int(MupixConfig* /*mupix config*/,int /*ASIC #*/)> func)
{
	INT status = DB_SUCCESS;


    // TODO: Change to odbxx
    //Retrieve number of ASICs
    odb nasics(prefix + "/Settings/ASICs/Global/Num asics");
    
    odb daq(prefix + "/Settings/Daq");

	//Iterate over ASICs
    // TODO: Get that back!
	/*for(unsigned int asic = 0; asic < nasics; ++asic) {

        if (daq["mask"][asic])
            continue;

		MupixConfig config;
		config.reset();
		char set_str[255];
		int status, size;
		// structs from ODB
		
		HNDLE hCHIPDACS;
        MUPIX_BIASDACS mt;
        sprintf(set_str, "%s/Settings/BIASDACS/%i", prefix, asic);
		printf("mupix_midasodb: Mapping ODB to Config for %s, asic %d: Using key %s\n",prefix,asic, set_str);
		status = db_find_key(db_rootentry, 0, set_str, &hCHIPDACS);
		if(status != DB_SUCCESS) {
			cm_msg(MERROR, "mupix::midasODB::MapMupixConfigFromDB", "Cannot find key %s", set_str);
		}
		size = sizeof(mt);
		status = db_get_record(db_rootentry, hCHIPDACS, &mt, &size, 0);
		if(status != DB_SUCCESS) {
			cm_msg(MERROR, "mupix::midasODB::MapMupixConfigFromDB", "Cannot retrieve settings");
		}
        config.Parse_BiasDACs_from_struct(mt);

        HNDLE hDIGIROWDACS;
        MUPIX_CONFDACS mdr;
        sprintf(set_str, "%s/Settings/CONFDACS/%i", prefix, asic);
        printf("mupix_midasodb: Mapping ODB to Config for %s, asic %d: Using key %s\n",prefix,asic, set_str);
        status = db_find_key(db_rootentry, 0, set_str, &hDIGIROWDACS);
        if(status != DB_SUCCESS) {
            cm_msg(MERROR, "mupix::midasODB::MapMupixConfigFromDB", "Cannot find key %s", set_str);
        }
        size = sizeof(mdr);
        status = db_get_record(db_rootentry, hDIGIROWDACS, &mdr, &size, 0);
        if(status != DB_SUCCESS) {
            cm_msg(MERROR, "mupix::midasODB::MapMupixConfigFromDB", "Cannot retrieve settings");
        }
        config.Parse_ConfDACs_from_struct(mdr);

        HNDLE hCOLDACS;
        MUPIX_VDACS mdc;
        sprintf(set_str, "%s/Settings/VDACS/%i", prefix, asic);
        printf("mupix_midasodb: Mapping ODB to Config for %s, asic %d: Using key %s\n",prefix,asic, set_str);
        status = db_find_key(db_rootentry, 0, set_str, &hCOLDACS);
        if(status != DB_SUCCESS) {
            cm_msg(MERROR, "mupix::midasODB::MapMupixConfigFromDB", "Cannot find key %s", set_str);
        }
        size = sizeof(mdc);
        status = db_get_record(db_rootentry, hCOLDACS, &mdc, &size, 0);
        if(status != DB_SUCCESS) {
            cm_msg(MERROR, "mupix::midasODB::MapMupixConfigFromDB", "Cannot retrieve settings");
        }
        config.Parse_VDACs_from_struct(mdc);

        //note: this needs to be passed as pointer, otherwise there is a memory corruption after exiting the lambda
		status=func(&config,asic);
		if (status != SUCCESS) break;
	}*/
	return status;
}


} } // namespace mupix::midasODB
