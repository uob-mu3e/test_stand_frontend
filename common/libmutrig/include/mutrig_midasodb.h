#ifndef MUTRIG_MIDASODB_H
#define MUTRIG_MIDASODB_H

#include <string>
#include <vector>
#include <sstream>
#include <functional>
#include <midas.h>
#include "mutrig_config.h"
#include "mutrig_MIDAS_config.h"

namespace mudaq { namespace mutrig {

/**
 * Mutrig configuration mapping between MIDAS ODB and ASIC bitpattern
 * TODO: do we need a class here?
 */

namespace midasODB {

//Setup ODB tree for Mutrig-based detectors.
//The tree will be built under e.g. prefix=/Equipment/SciFi
// /Equipment/SciFi/Daq
// /Equipment/SciFi/ASICs/Global
// /Equipment/SciFi/ASICs/%d/TDC
// /Equipment/SciFi/ASICs/%d/Channels/%d/Conf
//Relies on {prefix}/ASICs/Global/Num asics to build the tree of the right size
int setup_db(HNDLE& hDB, const char* prefix);

//Map ODB structure under prefix (e.g. /Equipment/SciFi) to a Config instance (i.e. build the configuration pattern) for this asic.
//Returns configuration class holding the pattern.
mudaq::mutrig::Config MapConfigFromDB(HNDLE& db_rootentry, const char* prefix, int asic);

//Foreach loop over all asics under this prefix. Call with a lambda function,
//e.g. midasODB::MapForEach(hDB, "/Equipment/SciFi",[mudaqdev_ptr](Config c,int asic){mudaqdev_ptr->ConfigureAsic(c,asic);});
//Function must return SUCCESS, otherwise loop is stopped.
int MapForEach(HNDLE& db_rootentry, const char* prefix, std::function<int(mudaq::mutrig::Config* /*mutrig config*/,int /*ASIC #*/)> func);

} } } // namespace mudaq::mutrig::midasODB

#endif // MUTRIG_MIDASODB_H
