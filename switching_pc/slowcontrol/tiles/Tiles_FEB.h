/********************************************************************\

  Name:         Tiles_FEB.h
  Created by:   Konrad Briggl

Contents:       Class to alter settings on a Tiles-FE-FPGA. Derives from MutrigFEB, thus here only tiles-specific methods would be defined.

\********************************************************************/

#ifndef TILES_FEB_ACCESS_H
#define TILES_FEB_ACCESS_H

#include "midas.h"
#include "FEBSlowcontrolInterface.h"
#include "Mutrig_FEB.h"
#include "odbxx.h"
using midas::odb;

class TilesFEB : public MutrigFEB{
   public:
      TilesFEB(const TilesFEB&)=delete;
      TilesFEB(FEBSlowcontrolInterface & feb_sc_,
               const vector<mappedFEB> & febs_,
               const uint64_t & febmask_,
               const char* equipment_name_,
               const char* odb_prefix_,
               const uint8_t SB_number_)
        :
    MutrigFEB(feb_sc_, febs_, febmask_, equipment_name_, odb_prefix_, SB_number_){}


      //Mapping from ASIC number to FPGA_ID and ASIC_ID
      virtual uint16_t FPGAid_from_ID(int asic) const;
      virtual uint16_t ASICid_from_ID(int asic) const;
      virtual uint16_t GetModulesPerFEB() const {return 1;}
      virtual uint16_t GetASICSPerModule() const {return 4;}
      //Return typeID for building FEB ID map
      virtual FEBTYPE  GetTypeID() const {return FEBTYPE::Tile;}

      //MIDAS callback for all ___ Tiles specific ___ setters.
      // Made static and using the user data argument as "this" to ease binding
      // to C-style midas-callbacks
      static void on_tiles_settings_changed(odb o, void * userdata);

};//class TilesFEB

#endif // TILES_FEB_ACCESS_H