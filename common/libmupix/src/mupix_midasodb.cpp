#include <cstring>
#include <iostream>
#include <iomanip>
//#include <mupix_config.h>
#include "midas.h"
//#include "experim.h"
#include "mupix_MIDAS_config.h"
#include "mupix_midasodb.h"
#include "odbxx.h"
using midas::odb;

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
        cm_msg(MINFO,"mupix_midasodb::setup_db","Number of ASICs is 0, will not continue to build DB. Consider to delete ODB subtree %s",prefix);
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

    /* Equipment/Pixel/Settings/Chipdacs (structure defined in mupix_MIDAS_config.h) */
    
    // # nasics 
    auto settings_chipdacs = MUPIX_CHIPDACS_SETTINGS;
    auto settings_chipdacs2 = MUPIX_CHIPDACS2_SETTINGS;
    auto settings_voltagedacs = MUPIX_VOLTAGEDACS_SETTINGS;
    
    // # nasics * nrows
    auto settings_digital_rowdacs = MUPIX_DIGIROWDACS_SETTINGS;
    auto settings_rowdacs = MUPIX_ROWDACS_SETTINGS;
    
    // # nasics * ncols
    auto settings_coldacs = MUPIX_COLDACS_SETTINGS;
    
    // # boards
    auto settings_boarddacs = MUPIX_BOARDDACS_SETTINGS;

    // get values from global odb instance
    nasics = settings_asics_global["Num asics"];
    unsigned int nboards = settings_asics_global["Num boards"];
    unsigned int nrows = settings_asics_global["Num rows"];
    unsigned int ncols = settings_asics_global["Num cols"];
        
    for(unsigned int i = 0; i < nasics; ++i) {
        sprintf(set_str, "%s/Settings/Chipdacs/%u", prefix, i);
        settings_chipdacs.connect(set_str, true);
        
        sprintf(set_str, "%s/Settings/Chipdacs2/%u", prefix, i);
        settings_chipdacs2.connect(set_str, true);
        
        sprintf(set_str, "%s/Settings/Voltagedacs/%u", prefix, i);
        settings_voltagedacs.connect(set_str, true);

        for (unsigned int row = 0; row < nrows; ++row) {
            sprintf(set_str, "%s/Settings/DigitalRowdacs/%u/row_%u", prefix, i, row);
            settings_digital_rowdacs.connect(set_str, true);
            
            sprintf(set_str, "%s/Settings/Rowdacs/%u/row_%u", prefix, i, row);
            settings_rowdacs.connect(set_str, true);
        }

        for (unsigned int col = 0; col < ncols; ++col) {
            sprintf(set_str, "%s/Settings/Coldacs/%u/col_%u", prefix, i, col);
            settings_coldacs.connect(set_str, true);
        }
    }

    //TODO: mask bits here?

    for(unsigned int i = 0; i < nboards; ++i) {
        sprintf(set_str, "%s/Settings/Boarddacs/%u", prefix, i);
        settings_boarddacs.connect(set_str, true);
    }

    return DB_SUCCESS;
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
	//Iterate over ASICs
	for(unsigned int asic = 0; asic < nasics; ++asic) {
		MupixConfig config;
		config.reset();
		char set_str[255];
		int status, size;
		// structs from ODB
		
		HNDLE hCHIPDACS;
		MUPIX_CHIPDACS mt;
		sprintf(set_str, "%s/Settings/Chipdacs/%i", prefix, asic);
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
		config.Parse_ChipDACs_from_struct(mt);

        HNDLE hDIGIROWDACS;
        MUPIX_DIGIROWDACS mdr;
        for (unsigned int row = 0; row < nrow; ++row) {
            sprintf(set_str, "%s/Settings/DigitalRowdacs/%i/row_%u", prefix, asic, row);
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
            config.Parse_DigiRowDACs_from_struct(mdr, row);
        }

        HNDLE hCOLDACS;
        MUPIX_COLDACS mdc;
        for (unsigned int col = 0; col < ncol; ++col) {
            sprintf(set_str, "%s/Settings/Coldacs/%i/col_%u", prefix, asic, col);
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
            config.Parse_ColDACs_from_struct(mdc, col);
        }

        HNDLE hROWDACS;
        MUPIX_ROWDACS mdro;
        for (unsigned int row = 0; row < nrow; ++row) {
            sprintf(set_str, "%s/Settings/Rowdacs/%i/row_%u", prefix, asic, row);
            printf("mupix_midasodb: Mapping ODB to Config for %s, asic %d: Using key %s\n",prefix,asic, set_str);
            status = db_find_key(db_rootentry, 0, set_str, &hROWDACS);
            if(status != DB_SUCCESS) {
                cm_msg(MERROR, "mupix::midasODB::MapMupixConfigFromDB", "Cannot find key %s", set_str);
            }
            size = sizeof(mdro);
            status = db_get_record(db_rootentry, hROWDACS, &mdro, &size, 0);
            if(status != DB_SUCCESS) {
                cm_msg(MERROR, "mupix::midasODB::MapMupixConfigFromDB", "Cannot retrieve settings");
            }
            config.Parse_RowDACs_from_struct(mdro, row);
        }

        HNDLE hCHIPDACS2;
        MUPIX_CHIPDACS2 mt2;
        sprintf(set_str, "%s/Settings/Chipdacs2/%i", prefix, asic);
        printf("mupix_midasodb: Mapping ODB to Config for %s, asic %d: Using key %s\n",prefix,asic, set_str);
        status = db_find_key(db_rootentry, 0, set_str, &hCHIPDACS2);
        if(status != DB_SUCCESS) {
            cm_msg(MERROR, "mupix::midasODB::MapMupixConfigFromDB", "Cannot find key %s", set_str);
        }
        size = sizeof(mt2);
        status = db_get_record(db_rootentry, hCHIPDACS2, &mt2, &size, 0);
        if(status != DB_SUCCESS) {
            cm_msg(MERROR, "mupix::midasODB::MapMupixConfigFromDB", "Cannot retrieve settings");
        }
        config.Parse_ChipDACs2_from_struct(mt2);

        HNDLE hVOLTAGEDACS;
        MUPIX_VOLTAGEDACS mv;
        sprintf(set_str, "%s/Settings/Voltagedacs/%i", prefix, asic);
        printf("mupix_midasodb: Mapping ODB to Config for %s, asic %d: Using key %s\n",prefix,asic, set_str);
        status = db_find_key(db_rootentry, 0, set_str, &hVOLTAGEDACS);
        if(status != DB_SUCCESS) {
            cm_msg(MERROR, "mupix::midasODB::MapMupixConfigFromDB", "Cannot find key %s", set_str);
        }
        size = sizeof(mv);
        status = db_get_record(db_rootentry, hVOLTAGEDACS, &mv, &size, 0);
        if(status != DB_SUCCESS) {
            cm_msg(MERROR, "mupix::midasODB::MapMupixConfigFromDB", "Cannot retrieve settings");
        }
        config.Parse_VoltageDACs_from_struct(mv);

        //note: this needs to be passed as pointer, otherwise there is a memory corruption after exiting the lambda
		status=func(&config,asic);
		if (status != SUCCESS) break;
	}
	return status;
}

int MapForEachBOARD(HNDLE& db_rootentry, const char* prefix, std::function<int(MupixBoardConfig* /*mupix config*/,int /*ASIC #*/)> func)
{
	INT status = DB_SUCCESS;
	char set_str[255];

	//Retrieve number of boards
	unsigned int nboards;
	int size = sizeof(nboards);
    sprintf(set_str, "%s/Settings/ASICs/Global/Num boards", prefix);
    status=db_get_value(db_rootentry, 0, set_str, &nboards, &size, TID_INT, 0);
	if (status != DB_SUCCESS) {
		cm_msg(MINFO,"mupix::midasODB::MapForEach", "Key %s not found", set_str);
		return status;
	}
	//Iterate over boards
	for(unsigned int board = 0; board < nboards; ++board) {
    		printf("mupix_midasodb: Mapping %s, board %d\n",prefix, board);
		MupixBoardConfig config;
		config.reset();
		char set_str[255];
		int status, size;
		// structs from ODB
		
		HNDLE hBOARDDACS;
		MUPIX_BOARDDACS mt;
		sprintf(set_str, "%s/Settings/Boarddacs/%i", prefix, board);
		printf("mupix_midasodb: Mapping ODB to Config for %s, board %d: Using key %s\n",prefix, board, set_str);
		status = db_find_key(db_rootentry, 0, set_str, &hBOARDDACS);
		if(status != DB_SUCCESS) {
			cm_msg(MERROR, "mupix::midasODB::MapBoardConfigFromDB", "Cannot find key %s", set_str);
		}
		size = sizeof(mt);
		status = db_get_record(db_rootentry, hBOARDDACS, &mt, &size, 0);
		if(status != DB_SUCCESS) {
			cm_msg(MERROR, "mupix::midasODB::MapBoardConfigFromDB", "Cannot retrieve settings");
		}
		config.Parse_BoardDACs_from_struct(mt);

		//note: this needs to be passed as pointer, otherwise there is a memory corruption after exiting the lambda
		status=func(&config,board);
		if (status != SUCCESS) break;
	}
	return status;
}


} } // namespace mupix::midasODB
