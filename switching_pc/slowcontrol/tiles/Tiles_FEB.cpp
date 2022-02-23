
/********************************************************************\

  Name:         Tiles_FEB.h
  Created by:   Konrad Briggl

Contents:       Class to alter settings on a Tiles-FE-FPGA. Derives from MutrigFEB, thus here only Tiles-specific methods would be defined.

\********************************************************************/
#include "Tiles_FEB.h"
#include "midas.h"
#include "odbxx.h"
#include "mfe.h" //for set_equipment_status
#include "../include/feb.h"

using midas::odb;
using namespace mu3e::daq;

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

//Read all power monitor readings from FEB/TMB, store in subtree $odb_prefix/Variables/tmb_current , tmb_voltage
//Parameter FPGA_ID refers to global numbering, i.e. before mapping
int TilesFEB::ReadBackTMBPower(mappedFEB & FEB){

   if(!FEB.IsScEnabled()) return SUCCESS; //skip disabled fibers
   if(FEB.SB_Number()!= SB_number) return SUCCESS; //skip commands not for this SB

   //issue readout on TMB
    auto rpc_ret = feb_sc.FEBsc_NiosRPC(FEB, feb::CMD_TILE_POWERMONITORS_READ, {});
   //retrieve results
   vector<uint32_t> val(rpc_ret*5*3);
   feb_sc.FEB_read(FEB, FEBSlowcontrolInterface::OFFSETS::FEBsc_RPC_DATAOFFSET,
                   val);

   return SUCCESS;
}

//Read all sipm matrix temperatures from FEB/TMB, store in subtree $odb_prefix/Variables/matrix_temperatures
//Parameter FPGA_ID refers to global numbering, i.e. before mapping
int TilesFEB::ReadBackMatrixTemperatures(mappedFEB & FEB){

   if(!FEB.IsScEnabled()) return SUCCESS; //skip disabled fibers
   if(FEB.SB_Number()!= SB_number) return SUCCESS; //skip commands not for this SB

   //issue readout on TMB
    auto rpc_ret = feb_sc.FEBsc_NiosRPC(FEB, feb::CMD_TILE_TEMPERATURES_READ, {});
   //retrieve results
   vector<uint32_t> val(rpc_ret*5*3);
   feb_sc.FEB_read(FEB, FEBSlowcontrolInterface::OFFSETS::FEBsc_RPC_DATAOFFSET,
                   val);

   return SUCCESS; 
}

