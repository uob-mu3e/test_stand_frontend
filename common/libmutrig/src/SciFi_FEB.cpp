
/********************************************************************\

  Name:         SciFi_FEB.h
  Created by:   Konrad Briggl

Contents:       Class to alter settings on a SciFi-FE-FPGA. Derives from MutrigFEB, thus here only scifi-specific methods would be defined.

\********************************************************************/
#include "SciFi_FEB.h"
#include "midas.h"
#include "mfe.h" //for set_equipment_status

SciFiFEB* SciFiFEB::m_instance=NULL;

//Mapping to physical ports of switching board.
uint8_t SciFiFEB::FPGAid_from_ID(int asic){return 0;}//return asic/4;}
uint8_t SciFiFEB::ASICid_from_ID(int asic){return asic;}//return asic%4;}



//MIDAS callback function for FEB register Setter functions
void SciFiFEB::on_scifi_settings_changed(HNDLE hDB, HNDLE hKey, INT, void * userdata)
{
   SciFiFEB* _this=static_cast<SciFiFEB*>(userdata);
   KEY key;
   db_get_key(hDB, hKey, &key);
   printf("SciFiFEB::on_settings_changed(%s)\n",key.name);
}


