#include <cstring>
#include <iostream>
#include <iomanip>

#include "midas.h"
#include "mupix_MIDAS_config.h"
#include "mupix_midasodb.h"
#include "odbxx.h"
#include "link_constants.h"
#include "mu3ebanks.h"
using midas::odb;
using namespace std;
using namespace mu3ebanks;

namespace mupix { namespace midasODB {


int setup_db(std::string prefix, int switch_id, bool write_defaults = true){
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

    // get total number of FEBs
    unsigned int nasics = settings_asics_global["Num asics"];
    unsigned int nFEBs = settings_asics_global["Num boards"];

    if(nasics==0){
        cm_msg(MINFO,"mupix_midasodb::setup_db","Number of Mupixes is 0, will not continue to build DB. Consider to delete ODB subtree %s",prefix.c_str());
        return DB_SUCCESS;
    }

    /* Add [prefix]/Daq (structure defined in mupix_MIDAS_config.h) */
    //TODO: if we have more than one FE-FPGA, there might be more than one DAQ class.
    auto settings_daq = MUPIX_DAQ_SETTINGS;
    settings_daq.connect(prefix + "/Settings/Daq", write_defaults);
    
    // create watch on MuPixFEB
    settings_daq.watch(MupixFEB::on_settings_changed);

    // set all dac values per asic
    auto settings_biasdacs = MUPIX_BIASDACS_SETTINGS;
    auto settings_confdacs = MUPIX_CONFDACS_SETTINGS;
    auto settings_vdacs = MUPIX_VDACS_SETTINGS;
    auto settings_tdacs = MUPIX_TDACS_SETTINGS;
        
    for(unsigned int i = 0; i < nasics; ++i) {
        settings_biasdacs.connect(prefix +  "/Settings/BIASDACS/" + to_string(i), write_defaults);

        settings_confdacs.connect(prefix +  "/Settings/CONFDACS/" + to_string(i), write_defaults);

        settings_vdacs.connect(prefix +  "/Settings/VDACS/" + to_string(i), write_defaults);

        settings_tdacs.connect(prefix +  "/Settings/TDACS/" + to_string(i), write_defaults);
    }

    // set all tdac values per FEB
    auto settings_febs = MUPIX_FEB_SETTINGS;

    for(unsigned int i = 0; i < nFEBs; ++i) {
        settings_febs.connect(prefix +  "/Settings/FEBS/" + to_string(i), write_defaults);
    }

    // PSLS Bank setup
    /* Default values for /Equipment/Mupix/Settings */
    // TODO: Get rid of N_FEBS_MUPIX_INT_2021
    odb settings = {
        {pslsnames[switch_id].c_str(), std::array<std::string, per_fe_PSLS_size*N_FEBS_MUPIX_INT_2021>{}}
    };

    // TODO: why do I have to connect here? In switch_fe.cpp we do first the naming and than we connect 
    // (NB: I have the same problem in the switch FE at some point)
    settings.connect(prefix + "/Settings", write_defaults);

    create_psls_names_in_odb(settings, switch_id, N_FEBS_MUPIX_INT_2021);

    settings.connect(prefix + "/Settings", write_defaults=write_defaults);

    /* Default values for /Equipment/Mupix/Variables */
    odb variables = {
        {psls[switch_id].c_str(), std::array<int, per_fe_PSLS_size*N_FEBS_MUPIX_INT_2021>{}}
    };

    variables.connect(prefix + "/Variables", write_defaults=write_defaults);

    return DB_SUCCESS;
}


int MapForEachASIC(std::string pixelprefix, std::function<int(MupixConfig* /*mupix config*/,int /*ASIC #*/)> func)
{
	INT status = DB_SUCCESS;


    //Retrieve number of ASICs
    odb nasics(pixelprefix + "/Settings/ASICs/Global/Num asics");
    
    odb daq(pixelprefix + "/Settings/Daq");

	//Iterate over ASICs
    // TODO: Get that back!
	for(unsigned int asic = 0; asic < nasics; ++asic) {

        if (daq["mask"][asic])
            continue;

		MupixConfig config;
		config.reset();
		//har set_str[255];
		//int status, size;
		// structs from ODB
        odb biasdacs(pixelprefix + "/Settings/BIASDACS/" + std::to_string(asic));
        config.Parse_BiasDACs_from_odb(biasdacs);

        odb confdacs(pixelprefix + "/Settings/CONFDACS/" + std::to_string(asic));
        config.Parse_ConfDACs_from_odb(confdacs);

        odb coldacs(pixelprefix +"/Settings/VDACS/" + std::to_string(asic));
        config.Parse_VDACs_from_odb(coldacs);

        //note: this needs to be passed as pointer, otherwise there is a memory corruption after exiting the lambda
		status=func(&config, asic);
		if (status != SUCCESS) break;
	}
	return status;
}


} } // namespace mupix::midasODB
