
/********************************************************************\

  Name:         SciFi_FEB.h
  Created by:   Konrad Briggl

Contents:       Class to alter settings on a SciFi-FE-FPGA. Derives from MutrigFEB, thus here only scifi-specific methods would be defined.

\********************************************************************/
#include "SciFi_FEB.h"
#include "midas.h"
#include "odbxx.h"
#include "mfe.h" //for set_equipment_status

using midas::odb;

//Mapping to physical ports of switching board.
uint16_t SciFiFEB::FPGAid_from_ID(int asic) const {return asic/GetASICSPerModule() + 0;}
uint16_t SciFiFEB::ASICid_from_ID(int asic) const {return asic%GetASICSPerModule() + 0;}
