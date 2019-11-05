#include <cstring>
#include <iostream>
#include <iomanip>
#include <mutrig_config.h>
#include "midas.h"
//#include "experim.h"
#include "mutrig_MIDAS_config.h"
#include "mutrig_midasodb.h"
namespace mutrig { namespace midasODB {
//#ifdef DEBUG_VERBOSE
//#define ddprintf(args...) printf(args)
//#else
//#define ddprintf(args...)
//#endif


int setup_db(HNDLE& hDB, const char* prefix, SciFiFEB* FEB_interface, bool init_FEB){
    /* Book Setting space */

    HNDLE hTmp;
    INT status = DB_SUCCESS;
    char set_str[255];

    /* Map Equipment/SciFi/Daq (structure defined in mutrig_MIDAS_config.h) */
    //TODO: if we have more than one FE-FPGA, there might be more than one DAQ class.
    MUTRIG_DAQ_STR(mutrig_daq);         // global settings for daq/fpga
    sprintf(set_str, "%s/Settings/Daq", prefix);
    status = db_create_record(hDB, 0, set_str, strcomb(mutrig_daq));
    status = db_find_key (hDB, 0, set_str, &hTmp);
    if (status != DB_SUCCESS) {
        cm_msg(MINFO,"frontend_init", "Key %s not found", set_str);
        return status;   
    }
//    HNDLE key_tmp;
//    if(nasics >0){
//        sprintf(set_str, "%s/Daq/mask", prefix);
//        db_find_key(hDB, 0, set_str, &key_tmp);
//        db_set_num_values(hDB, key_tmp, nasics);
//    }

    //open hot link
    db_watch(hDB, hTmp, &SciFiFEB::on_settings_changed, FEB_interface);
    //init all values on FEB
    if(init_FEB){
        INT ival;
        BOOL bval;
	INT bsize=sizeof(bval);
	INT isize=sizeof(ival);

        sprintf(set_str, "%s/Settings/Daq/dummy_config", prefix);
        db_find_key(hDB, 0, set_str, &hTmp);
        db_get_data(hDB,hTmp,&bval,&bsize,TID_BOOL);
        FEB_interface->setDummyConfig(SciFiFEB::FPGA_broadcast_ID,bval);

        sprintf(set_str, "%s/Settings/Daq/dummy_data", prefix);
        db_find_key(hDB, 0, set_str, &hTmp);
        db_get_data(hDB,hTmp,&bval,&bsize,TID_BOOL);
        FEB_interface->setDummyData_Enable(SciFiFEB::FPGA_broadcast_ID,bval);

        sprintf(set_str, "%s/Settings/daq/dummy_data_fast", prefix);
        db_find_key(hDB, 0, set_str, &hTmp);
        db_get_data(hDB,hTmp,&bval,&bsize,TID_BOOL);
        FEB_interface->setDummyData_Fast(SciFiFEB::FPGA_broadcast_ID,bval);

        sprintf(set_str, "%s/Settings/Daq/dummy_data_n", prefix);
        db_find_key(hDB, 0, set_str, &hTmp);
        db_get_data(hDB,hTmp,&ival,&isize,TID_INT);
        FEB_interface->setDummyData_Count(SciFiFEB::FPGA_broadcast_ID,ival);

        sprintf(set_str, "%s/Settings/Daq/prbs_decode_bypass", prefix);
        db_find_key(hDB, 0, set_str, &hTmp);
        db_get_data(hDB,hTmp,&bval,&bsize,TID_BOOL);
        FEB_interface->setPRBSDecoder(SciFiFEB::FPGA_broadcast_ID,bval);

	BOOL barray[16];
	INT  barraysize=sizeof(barray);
        sprintf(set_str, "%s/Settings/Daq/mask", prefix);
        db_find_key(hDB, 0, set_str, &hTmp);
        db_get_data(hDB,hTmp,barray,&barraysize,TID_BOOL);
	for(int i=0;i<16;i++)
		FEB_interface->setMask(i,barray[i]);
    }

    /* Map Equipment/SciFi/ASICs/Global (structure defined in mutrig_MIDAS_config.h) */
    //TODO some globals should be per asic
    MUTRIG_GLOBAL_STR(mutrig_global);   // global mutrig settings
    sprintf(set_str, "%s/Settings/ASICs/Global", prefix);
    //ddprintf("mutrig_midasodb: adding struct %s\n",set_str);
    status = db_create_record(hDB, 0, set_str, strcomb(mutrig_global));
    status = db_find_key (hDB, 0, set_str, &hTmp);
    if (status != DB_SUCCESS) {
        cm_msg(MINFO,"frontend_init", "Key %s not found", set_str);
        return status;
    }

   //Get predefined number of asics from ODB
    unsigned int nasics;
    int size = sizeof(nasics);
    sprintf(set_str, "%s/Settings/ASICs/Global/Num asics", prefix);
    db_get_value(hDB, 0, set_str, &nasics, &size, TID_INT, 0);

    //ddprintf("mutrig_midasodb: number of asics set to %d\n",nasics);

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

int MapForEach(HNDLE& db_rootentry, const char* prefix, std::function<int(MutrigConfig* /*mutrig config*/,int /*ASIC #*/)> func)
{
	INT status = DB_SUCCESS;
	char set_str[255];

	//Retrieve number of ASICs
	unsigned int nasics;
	int size = sizeof(nasics);
	sprintf(set_str, "%s/Settings/ASICs/Global/Num asics", prefix);
	status=db_get_value(db_rootentry, 0, set_str, &nasics, &size, TID_INT, 0);
	if (status != DB_SUCCESS) {
		cm_msg(MINFO,"mutrig::midasODB::MapForEach", "Key %s not found", set_str);
		return status;
	}
	//Iterate over ASICs
	for(unsigned int asic = 0; asic < nasics; ++asic) {
    		//ddprintf("mutrig_midasodb: Mapping %s, asic %d\n",prefix, asic);
		MutrigConfig config(MapConfigFromDB(db_rootentry,prefix,asic));
		//note: this needs to be passed as pointer, otherwise there is a memory corruption after exiting the lambda
		status=func(&config,asic);
		if (status != SUCCESS) break;
	}
	return status;
}
} } // namespace mutrig::midasODB
