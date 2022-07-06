#ifndef MUTRIG_MIDASODB_H
#define MUTRIG_MIDASODB_H

#include <string>
#include <vector>
#include <sstream>
#include <functional>
#include <midas.h>
#include "MutrigConfig.h"
#include "mutrig_MIDAS_config.h"
#include "Mutrig_FEB.h"

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
int setup_db(std::string prefix, MutrigFEB & FEB_inteface, uint32_t nasics, uint32_t nModules, uint32_t nAsicsPerFeb, bool write_defaults);

//Map ODB structure under prefix (e.g. /Equipment/SciFi) to a Config instance (i.e. build the configuration pattern) for this asic.
//Returns configuration class holding the pattern.
MutrigConfig MapConfigFromDB(odb settings_asics, int asic);

} } // namespace mutrig::midasODB

#endif // MUTRIG_MIDASODB_H
