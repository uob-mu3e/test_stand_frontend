#ifndef MUPIX_MIDASODB_H
#define MUPIX_MIDASODB_H

#include <string>
#include <vector>
#include <sstream>
#include <functional>
#include <midas.h>
#include "mupix_config.h"
#include "mupix_MIDAS_config.h"
#include "mupix_FEB.h"

namespace mupix {

/**
 * Mutrig configuration mapping between MIDAS ODB and ASIC bitpattern
 * TODO: do we need a class here?
 */

namespace midasODB {

//Setup ODB tree for Mutrig-based detectors.
//The tree will be built under e.g. prefix=/Equipment/Mupix
// /Equipment/Mupix/Settings/Daq
// /Equipment/Mupix/Settings/ASICs/Global
// /Equipment/Mupix/Settings/ASICs/%d/TDC
// /Equipment/Mupix/Settings/ASICs/%d/Channels/%d/Conf
//Relies on {prefix}/Settings/ASICs/Global/Num asics to build the tree of the right size
//If init_FEB is set, the registers on the FEB-FPGA are initialized
int setup_db(const char* prefix,MupixFEB* FEB_inteface,bool init_FEB);

//Foreach loop over all boards/asics under this prefix. Call with a lambda function,
//e.g. midasODB::MapForEach(hDB, "/Equipment/Mupix",[mudaqdev_ptr](Config c,int asic){mudaqdev_ptr->ConfigureAsic(c,asic);});
//Function must return SUCCESS, otherwise loop is stopped.
int MapForEachASIC(HNDLE& db_rootentry, const char* prefix, std::function<int(MupixConfig* /*mupix config*/,int /*ASIC #*/)> func);
int MapForEachBOARD(HNDLE& db_rootentry, const char* prefix, std::function<int(MupixBoardConfig* /*mupix config*/,int /*ASIC #*/)> func);

} } // namespace mupix::midasODB

#endif // MUPIX_MIDASODB_H
