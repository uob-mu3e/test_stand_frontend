
/********************************************************************\

  Name:         Tiles_FEB.h
  Created by:   Konrad Briggl

Contents:       Class to alter settings on a Tiles-FE-FPGA. Derives from MutrigFEB, thus here only Tiles-specific methods would be defined.

\********************************************************************/
#include "Tiles_FEB.h"
#include "midas.h"
#include "mfe.h" //for set_equipment_status

TilesFEB* TilesFEB::m_instance=NULL;

//Mapping to physical ports of switching board.
uint8_t TilesFEB::FPGAid_from_ID(int asic){return 0;}//return asic/4;}
uint8_t TilesFEB::ASICid_from_ID(int asic){return asic;}//return asic%4;}



//MIDAS callback function for FEB register Setter functions
void TilesFEB::on_tiles_settings_changed(HNDLE hDB, HNDLE hKey, INT, void * userdata)
{
   TilesFEB* _this=static_cast<TilesFEB*>(userdata);
   KEY key;
   db_get_key(hDB, hKey, &key);
   printf("TilesFEB::on_settings_changed(%s)\n",key.name);
}


