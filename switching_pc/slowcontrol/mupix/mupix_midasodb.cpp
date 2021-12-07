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


int setup_db(const char* prefix, MupixFEB* FEB_interface, bool init_FEB){
    /* Book Setting space */
    
    cm_msg(MINFO, "mupix_midasodb::setup_db", "Setting up odb");
    
    char set_str[255];

    /* Add [prefix]/ASICs/Global (structure defined in mutrig_MIDAS_config.h) */
    //TODO some globals should be per asic
    sprintf(set_str, "%s/Settings/ASICs/Global", prefix);
    auto settings_asics_global = MUPIX_GLOBAL_SETTINGS;
    // global mupix settings from mupix_MIDAS_config.h
    settings_asics_global.connect(set_str, true); 

    //Set number of ASICs, derived from mapping
    unsigned int nasics = FEB_interface->GetNumASICs();
    settings_asics_global["Num asics"] = nasics;
    // TODO why is this the same?
    settings_asics_global["Num boards"] = nasics;

    if(nasics==0){
        cm_msg(MINFO,"mupix_midasodb::setup_db","Number of Mupixes is 0, will not continue to build DB. Consider to delete ODB subtree %s",prefix);
    return DB_SUCCESS;
    }

    /* Add [prefix]/Daq (structure defined in mupix_MIDAS_config.h) */
    //TODO: if we have more than one FE-FPGA, there might be more than one DAQ class.
    sprintf(set_str, "%s/Settings/Daq", prefix);
    auto settings_daq = MUPIX_DAQ_SETTINGS;
    settings_daq.connect(set_str, true);
    
    // use lambda funciton for passing FEB_interface
    auto on_settings_changed_partial = 
        [&FEB_interface](odb o) { 
            return MupixFEB::on_settings_changed(
                o, FEB_interface
            );
        };
    settings_daq.watch(on_settings_changed_partial);

    //init all values on FEB
    if(init_FEB){
        BOOL bval;

        bval = settings_daq["dummy_config"];
        // TODO: do something here
        //FEB_interface->setDummyConfig(SciFiFEB::FPGA_broadcast_ID,bval); 
        
        bval = settings_daq["dummy_data"];
        // TODO: do something here
        //FEB_interface->setDummyData_Enable(SciFiFEB::FPGA_broadcast_ID,bval);
        
        // TODO: do something here
        // for(int i=0;i<16;i++)
        // FEB_interface->setMask(i,settings_daq["mask"]i]);
    }

    auto settings_biasdacs = MUPIX_BIASDACS_SETTINGS;
    auto settings_confdacs = MUPIX_CONFDACS_SETTINGS;
    auto settings_vdacs = MUPIX_VDACS_SETTINGS;
    auto settings_tdacs = MUPIX_TDACS_SETTINGS;
    

    nasics = settings_asics_global["Num asics"];
        
    for(unsigned int i = 0; i < nasics; ++i) {
        sprintf(set_str, "%s/Settings/BIASDACS/%u", prefix, i);
        settings_biasdacs.connect(set_str, true);
        
        sprintf(set_str, "%s/Settings/CONFDACS/%u", prefix, i);
        settings_confdacs.connect(set_str, true);
        
        sprintf(set_str, "%s/Settings/VDACS/%u", prefix, i);
        settings_vdacs.connect(set_str, true);
        
        sprintf(set_str, "%s/Settings/TDACS/%u", prefix, i);
        settings_tdacs.connect(set_str, true);
    }

    // PSLL Bank setup
    /* Default values for /Equipment/Mupix/Settings */
    odb settings = {
        {namestrPSLL.c_str(), std::array<std::string, per_fe_PSLL_size*N_FEBS_MUPIX_INT_2021*lvds_links_per_feb>{}}
    };

    // TODO: why do I have to connect here? In switch_fe.cpp we do first the naming and than we connect
    settings.connect(set_str, true);

    create_psll_names_in_odb(settings, N_FEBS_MUPIX_INT_2021, lvds_links_per_feb);

    sprintf(set_str, "%s/Settings", prefix);
    settings.connect(set_str, true);

    /* Default values for /Equipment/Mupix/Variables */
    odb variables = {
        {banknamePSLL.c_str(), std::array<int, per_fe_PSLL_size*N_FEBS_MUPIX_INT_2021*lvds_links_per_feb>{}}
    };

    sprintf(set_str, "%s/Variables", prefix);
    variables.connect(set_str, true);

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


int MapForEachASIC(HNDLE& db_rootentry, const char* prefix, std::function<int(MupixConfig* /*mupix config*/,int /*ASIC #*/)> func)
{
	INT status = DB_SUCCESS;
	char set_str[255];

    unsigned int nrow = 200;
    unsigned int ncol = 128; //TODO: somewhere global?

    //Retrieve number of ASICs
	INT nasics;
	int size = sizeof(nasics);
    sprintf(set_str, "%s/Settings/ASICs/Global/Num asics", prefix);
    status=db_get_value(db_rootentry, 0, set_str, &nasics, &size, TID_INT, 0);
	if (status != DB_SUCCESS) {
		cm_msg(MINFO,"mupix::midasODB::MapForEach", "Key %s not found", set_str);
		return status;
	}

    //Retrieve daq settings for mask
    HNDLE hDAQ;
    MUPIX_DAQ mdaq;
    sprintf(set_str, "%s/Settings/Daq", prefix);
    printf("mupix_midasodb: Mapping ODB to Config for %s: Using key %s\n",prefix, set_str);
    status = db_find_key(db_rootentry, 0, set_str, &hDAQ);
    if(status != DB_SUCCESS) {
        cm_msg(MERROR, "mupix::midasODB::MapMupixConfigFromDB", "Cannot find key %s", set_str);
    }
    size = sizeof(mdaq);
    status = db_get_record(db_rootentry, hDAQ, &mdaq, &size, 0);
    if(status != DB_SUCCESS) {
        cm_msg(MERROR, "mupix::midasODB::MapMupixConfigFromDB", "Cannot retrieve settings");
    }


	//Iterate over ASICs
	for(unsigned int asic = 0; asic < nasics; ++asic) {

        if (mdaq.mask[asic])
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
	}
	return status;
}


} } // namespace mupix::midasODB
