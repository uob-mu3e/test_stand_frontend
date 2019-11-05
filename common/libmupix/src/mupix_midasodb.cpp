#include <cstring>
#include <iostream>
#include <iomanip>
//#include <mupix_config.h>
#include "midas.h"
//#include "experim.h"
#include "mupix_MIDAS_config.h"
#include "mupix_midasodb.h"
namespace mupix { namespace midasODB {


int setup_db(HNDLE& hDB, const char* prefix, MupixFEB* FEB_interface, bool init_FEB){
    /* Book Setting space */

    HNDLE hTmp;
    INT status = DB_SUCCESS;
    char set_str[255];

    /* Map Equipment/Pixel/Daq (structure defined in mupix_MIDAS_config.h) */
    //TODO: if we have more than one FE-FPGA, there might be more than one DAQ class.
    MUPIX_DAQ_STR(mupix_daq);         // global settings for daq/fpga
    sprintf(set_str, "%s/Settings/Daq", prefix);
    status = db_create_record(hDB, 0, set_str, strcomb(mupix_daq));
    status = db_find_key (hDB, 0, set_str, &hTmp);
    if (status != DB_SUCCESS) {
        cm_msg(MINFO,"frontend_init", "Key %s not found", set_str);
        return status;   
    }
    //open hot link
    db_watch(hDB, hTmp, &MupixFEB::on_settings_changed, FEB_interface);

    //init all values on FEB
    if(init_FEB){
        INT ival;
        BOOL bval;
	INT bsize=sizeof(bval);

        sprintf(set_str, "%s/Settings/Daq/dummy_config", prefix);
        db_find_key(hDB, 0, set_str, &hTmp);
        db_get_data(hDB,hTmp,&bval,&bsize,TID_BOOL);
        //FEB_interface->setDummyConfig(SciFiFEB::FPGA_broadcast_ID,bval);

        sprintf(set_str, "%s/Settings/Daq/dummy_data", prefix);
        db_find_key(hDB, 0, set_str, &hTmp);
        db_get_data(hDB,hTmp,&bval,&bsize,TID_BOOL);
        //FEB_interface->setDummyData_Enable(SciFiFEB::FPGA_broadcast_ID,bval);

	BOOL barray[16];
	INT  barraysize=sizeof(barray);
        sprintf(set_str, "%s/Settings/Daq/mask", prefix);
        db_find_key(hDB, 0, set_str, &hTmp);
        db_get_data(hDB,hTmp,barray,&barraysize,TID_BOOL);
	//for(int i=0;i<16;i++)
		//FEB_interface->setMask(i,barray[i]);
    }

    unsigned int n;
    int size = sizeof(n);

    /* Equipment/Pixel/Settings/Boarddacs (structure defined in mupix_MIDAS_config.h) */
    //Get predefined number of boards from ODB
    sprintf(set_str, "%s/Settings/Global/Num boards", prefix);
    db_get_value(hDB, 0, set_str, &n, &size, TID_INT, 0);

    MUPIX_BOARDDACS_STR(mupix_boarddacs);
    for(unsigned int i = 0; i < n; ++i) {
        sprintf(set_str, "%s/Settings/Boarddacs/%u", prefix, i);
        status = db_create_record(hDB, 0, set_str, strcomb(mupix_boarddacs));
        status = db_find_key (hDB, 0, set_str, &hTmp);
        if (status != DB_SUCCESS) {
            cm_msg(MINFO,"frontend_init", "Key %s not found", set_str);
            return status;
        }
    }

    /* Equipment/Pixel/Settings/Chipdacs (structure defined in mupix_MIDAS_config.h) */
    //Get predefined number of asics from ODB
    sprintf(set_str, "%s/Settings/Global/Num asics", prefix);
    db_get_value(hDB, 0, set_str, &n, &size, TID_INT, 0);

    MUPIX_CHIPDACS_STR(mupix_chipdacs);
    for(unsigned int i = 0; i < n; ++i) {
        sprintf(set_str, "%s/Settings/Chipdacs/%u", prefix,i);
        status = db_create_record(hDB, 0, set_str, strcomb(mupix_chipdacs));
        status = db_find_key (hDB, 0, set_str, &hTmp);
        if (status != DB_SUCCESS) {
            cm_msg(MINFO,"frontend_init", "Key %s not found", set_str);
            return status;
        }
    }
    //TODO: mask bits here?
    return status;
}


int MapForEachASIC(HNDLE& db_rootentry, const char* prefix, std::function<int(MupixConfig* /*mupix config*/,int /*ASIC #*/)> func)
{
	INT status = DB_SUCCESS;
	char set_str[255];

	//Retrieve number of ASICs
	unsigned int nasics;
	int size = sizeof(nasics);
	sprintf(set_str, "%s/Settings/Global/Num asics", prefix);
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
	sprintf(set_str, "%s/Settings/Global/Num boards", prefix);
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
