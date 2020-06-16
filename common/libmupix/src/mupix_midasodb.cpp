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

    unsigned int nrow = 200;
    unsigned int ncol = 128; //TODO: somewhere global?
    char set_str[255];
    unsigned int nboards;

    /* Add [prefix]/ASICs/Global (structure defined in mutrig_MIDAS_config.h) */
    //TODO some globals should be per asic
    MUPIX_GLOBAL_STR(mupix_global);   // global mutrig settings
    sprintf(set_str, "%s/Settings/ASICs", prefix);
    odb settings_asics(set_str);
    settings_asics["/Global"] = strcomb(mupix_global);

    //Set number of ASICs, derived from mapping
    unsigned int nasics=FEB_interface->GetNumASICs();
    settings_asics["/Num asics"] = nasics;
    settings_asics["/Num boards"] = nasics;

    if(nasics==0){
        cm_msg(MINFO,"mutrig_midasodb::setup_db","Number of ASICs is 0, will not continue to build DB. Consider to delete ODB subtree %s",prefix);
    return DB_SUCCESS;
    }

    /* Add [prefix]/Daq (structure defined in mupix_MIDAS_config.h) */
    //TODO: if we have more than one FE-FPGA, there might be more than one DAQ class.
    MUPIX_DAQ_STR(mupix_daq);         // global settings for daq/fpga
    sprintf(set_str, "%s/Settings/Daq", prefix);
    odb settings_daq(set_str);
    settings_daq = strcomb(mupix_daq); 
    // use lambda funciton for passing FEB_interface
    auto on_settings_changed_partial = 
        [&FEB_interface](odb o) { return MupixFEB::on_settings_changed(o, FEB_interface);
        };


    //init all values on FEB
    if(init_FEB){
        BOOL bval;

        // get odb instance
        sprintf(set_str, "%s/Settings/Daq", prefix);
        odb settings_daq(set_str);

        bval = settings_daq["/dummy_config"];
        // TODO: do something here
        //FEB_interface->setDummyConfig(SciFiFEB::FPGA_broadcast_ID,bval); 
        
        bval = settings_daq["dummy_daq"];
        // TODO: do something here
        //FEB_interface->setDummyData_Enable(SciFiFEB::FPGA_broadcast_ID,bval);
        
        // TODO: do something here
        // for(int i=0;i<16;i++)
        // FEB_interface->setMask(i,settings_daq["mask"]i]);
    }

    /* Equipment/Pixel/Settings/Chipdacs (structure defined in mupix_MIDAS_config.h) */
    MUPIX_CHIPDACS_STR(mupix_chipdacs);
    MUPIX_DIGIROWDACS_STR(mupix_digirowdacs);
    MUPIX_COLDACS_STR(mupix_coldacs);
    MUPIX_ROWDACS_STR(mupix_rowdacs);
    MUPIX_CHIPDACS2_STR(mupix_chipdacs2);
    MUPIX_VOLTAGEDACS_STR(mupix_voltagedacs);

    // get odb instances
    sprintf(set_str, "%s/Settings/Chipdacs", prefix);
    odb settings_chipdacs(set_str);
    
    sprintf(set_str, "%s/Settings/DigitalRowdacs", prefix);
    odb settings_digital_rowdacs(set_str);
    
    sprintf(set_str, "%s/Settings/Coldacs", prefix);
    odb settings_coldacs(set_str);

    sprintf(set_str, "%s/Settings/Rowdacs", prefix); 
    odb settings_rowdacs(set_str);

    sprintf(set_str, "%s/Settings/Chipdacs2", prefix);
    odb settings_chipdacs2(set_str);

    sprintf(set_str, "%s/Settings/Voltagedacs", prefix);
    odb settings_voltagedacs(set_str);
    
    for(unsigned int i = 0; i < nasics; ++i) {
        sprintf(set_str, "/%u", i);
        settings_chipdacs[set_str] = strcomb(mupix_chipdacs);
        settings_chipdacs2[set_str] = strcomb(mupix_chipdacs2);
        settings_voltagedacs[set_str] = strcomb(mupix_chipdacs2);

        for (unsigned int row = 0; row < nrow; ++row) {
            sprintf(set_str, "/%u/row_%u", i, row);
            settings_digital_rowdacs[set_str] = strcomb(mupix_digirowdacs);
            settings_rowdacs[set_str] = strcomb(mupix_rowdacs);
        }

        for (unsigned int col = 0; col < ncol; ++col) {
            sprintf(set_str, "/%u/col_%u", i, col);
            settings_coldacs[set_str] = strcomb(mupix_coldacs);
        }
    }

    //TODO: mask bits here?

    /* Equipment/Pixel/Settings/Boarddacs (structure defined in mupix_MIDAS_config.h) */
    //Get predefined number of boards from ODB
    MUPIX_BOARDDACS_STR(mupix_boarddacs);

    // get nboards from odb instance
    sprintf(set_str, "%s/Settings/ASICs/Global", prefix);
    odb settings_asics_global(set_str);
    nboards = settings_asics_global["/Num boards"];

    sprintf(set_str, "%s/Settings/Boarddacs", prefix);
    odb settings_boarddacs(set_str);
    for(unsigned int i = 0; i < nboards; ++i) {
        sprintf(set_str, "/%u", i); 
        settings_boarddacs[set_str] = strcomb(mupix_boarddacs);
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
