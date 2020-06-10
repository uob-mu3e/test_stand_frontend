/********************************************************************\

  Name:         Tiles_FEB.h
  Created by:   Konrad Briggl

Contents:       Class to alter settings on a Tiles-FE-FPGA. Derives from MutrigFEB, thus here only tiles-specific methods would be defined.

\********************************************************************/

#ifndef TILES_FEB_ACCESS_H
#define TILES_FEB_ACCESS_H

#include "midas.h"
#include "mudaq_device_scifi.h"
#include "Mutrig_FEB.h"
#include "odbxx.h"
using midas::odb;

class TilesFEB : public MutrigFEB{
   private:
      static TilesFEB* m_instance; //signleton instance pointer: only one instance of TilesFEB
      TilesFEB(const TilesFEB&)=delete;
      TilesFEB(mudaq::MudaqDevice& mu, const char* equipment_name, const char* odb_prefix)
	:
	MutrigFEB(mu,equipment_name,odb_prefix)
        {
		RebuildFEBsMap();
        };
   public:
      static TilesFEB* Create(mudaq::MudaqDevice& mu, const char* equipment_name, const char* odb_prefix){printf("TilesFEB::Create(%s) as %s\n",odb_prefix,equipment_name);if(!m_instance) m_instance=new TilesFEB(mu,equipment_name,odb_prefix); return m_instance;};
      static TilesFEB* Instance(){return m_instance;};

      //Mapping from ASIC number to FPGA_ID and ASIC_ID
      virtual uint16_t FPGAid_from_ID(int asic);
      virtual uint16_t ASICid_from_ID(int asic);
      virtual uint16_t nModulesPerFEB(){return 1;}
      virtual uint16_t nAsicsPerModule(){return 4;}
      //Return typeID for building FEB ID map
      virtual FEBTYPE  GetTypeID(){return FEBTYPE::Tile;}

      //MIDAS callback for all ___ Tiles specific ___ setters. Made static and using the user data argument as "this" to ease binding to C-style midas-callbacks
      static void on_tiles_settings_changed(odb o, void * userdata);

};//class TilesFEB

#endif // TILES_FEB_ACCESS_H
