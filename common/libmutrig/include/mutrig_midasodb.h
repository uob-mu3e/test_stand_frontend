#ifndef MUTRIG_MIDASODB_H
#define MUTRIG_MIDASODB_H

#include <string>
#include <vector>
#include <sstream>
#include <functional>
#include <midas.h>
#include "mutrig_config.h"
#include "mutrig_MIDAS_config.h"
#include "SciFi_FEB.h"

namespace mutrig {

/**
 * Mutrig configuration mapping between MIDAS ODB and ASIC bitpattern
 * TODO: do we need a class here?
 */

namespace midasODB {

//Setup ODB tree for Mutrig-based detectors.
//The tree will be built under e.g. prefix=/Equipment/SciFi
// /Equipment/SciFi/Settings/Daq
// /Equipment/SciFi/Settings/ASICs/Global
// /Equipment/SciFi/Settings/ASICs/%d/TDC
// /Equipment/SciFi/Settings/ASICs/%d/Channels/%d/Conf
//Relies on {prefix}/Settings/ASICs/Global/Num asics to build the tree of the right size
//If init_FEB is set, the registers on the FEB-FPGA are initialized
int setup_db(HNDLE& hDB, const char* prefix,SciFiFEB* FEB_inteface,bool init_FEB);

//Map ODB structure under prefix (e.g. /Equipment/SciFi) to a Config instance (i.e. build the configuration pattern) for this asic.
//Returns configuration class holding the pattern.
MutrigConfig MapConfigFromDB(HNDLE& db_rootentry, const char* prefix, int asic);

//Foreach loop over all asics under this prefix. Call with a lambda function,
//e.g. midasODB::MapForEach(hDB, "/Equipment/SciFi",[mudaqdev_ptr](Config c,int asic){mudaqdev_ptr->ConfigureAsic(c,asic);});
//Function must return SUCCESS, otherwise loop is stopped.
int MapForEach(HNDLE& db_rootentry, const char* prefix, std::function<int(mutrig::MutrigConfig* /*mutrig config*/,int /*ASIC #*/)> func);

} } // namespace mutrig::midasODB

#endif // MUTRIG_MIDASODB_H
