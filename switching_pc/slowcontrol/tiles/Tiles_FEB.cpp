
/********************************************************************\

  Name:         Tiles_FEB.h
  Created by:   Konrad Briggl

Contents:       Class to alter settings on a Tiles-FE-FPGA. Derives from MutrigFEB, thus here only Tiles-specific methods would be defined.

\********************************************************************/
#include "Tiles_FEB.h"
#include "midas.h"
#include "odbxx.h"
#include "mfe.h" //for set_equipment_status

using midas::odb;

//Mapping to physical ports of switching board.
uint16_t TilesFEB::FPGAid_from_ID(int asic) const {return asic/GetASICSPerModule() + 0;}
uint16_t TilesFEB::ASICid_from_ID(int asic) const {return asic%GetASICSPerModule() + 0;}


//MIDAS callback function for FEB register Setter functions
void TilesFEB::on_tiles_settings_changed(odb o, void * userdata)
{
   TilesFEB* _this=static_cast<TilesFEB*>(userdata);
   std::string name = o.get_name();
   cm_msg(MINFO, "TilesFEB::on_settings_changed", "Setting changed (%s)", name.c_str());
}


