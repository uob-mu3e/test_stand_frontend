/********************************************************************\

  Name:         Mupix_FEB.h
  Created by:   Konrad Briggl

Contents:       Definition of functions to talk to a mupix-based FEB. Designed to no be derived into a SciFi_FEB and a Tiles_FEB class where subdetector specific actions are defined.
		Here: Definition of basic things for mupix-configuration & datapath settings

\********************************************************************/

#include "mupix_FEB.h"
#include "midas.h"
#include "odbxx.h"
#include "mfe.h" //for set_equipment_status
#include "odbxx.h"

#include "../include/feb.h"
using namespace mu3e::daq;

#include "mudaq_device.h"
#include "mupix_config.h"
#include "mupix_midasodb.h"
#include <thread>
#include <chrono>

using midas::odb;

#include "default_config_mupix.h" //TODO avoid this, reproduce configure routine from chip dacs

//offset for registers on nios SC memory
#define SC_REG_OFFSET 0xff60
#define FE_DUMMYCTRL_REG       (SC_REG_OFFSET+0x8)
#define FE_DPCTRL_REG          (SC_REG_OFFSET+0x9)
#define FE_SUBDET_RESET_REG    (SC_REG_OFFSET+0xa)
#define FE_SPIDATA_ADDR		0

//status flags from FEB
#define FEB_REPLY_SUCCESS 0
#define FEB_REPLY_ERROR   1

//Mapping to physical ports of switching board.
uint16_t MupixFEB::FPGAid_from_ID(int asic) const {return asic/12;}
uint16_t MupixFEB::ASICid_from_ID(int asic) const {return asic%12;}

uint16_t MupixFEB::GetNumASICs() const {return febs.size()*12;} //TODO: add parameter for number of asics per FEB, later more flexibility to have different number of sensors per FEB

void invert_datastream(uint32_t * datastream) {

}

//ASIC configuration:
//Configure all asics under prefix (e.g. prefix="/Equipment/Mupix")
int MupixFEB::ConfigureASICs(){
   printf("MupixFEB::ConfigureASICs()\n");
   cm_msg(MINFO, "MupixFEB" , "Configuring sensors under prefix %s/Settings/ASICs/", odb_prefix);
   int status = mupix::midasODB::MapForEachASIC(hDB,odb_prefix,[this](mupix::MupixConfig* config, int asic){
      uint32_t rpc_status;
      bool TDACsNotFound = false;
      int useTDACs = 0;
      short tDAC=0;
      u_int32_t rowRAM_addr=0;

      //mapping
      uint16_t SB_ID=febs[FPGAid_from_ID(asic)].SB_Number();
      uint16_t SP_ID=febs[FPGAid_from_ID(asic)].SB_Port();
      uint16_t FA_ID=ASICid_from_ID(asic);

      if(!febs[FPGAid_from_ID(asic)].IsScEnabled()){
          printf(" [skipped]\n");
          return FE_SUCCESS;
      }
      if(SB_ID!= SB_number){
          printf(" [skipped]\n");
          return FE_SUCCESS;
      } //TODO

      cm_msg(MINFO, "MupixFEB" ,
             "Configuring sensor %s/Settings/ASICs/%i/: Mapped to FEB%u -> SB%u.%u  ASIC #%d",
             odb_prefix,asic,FPGAid_from_ID(asic),SB_ID,SP_ID,FA_ID);

    // TODO: There is a lot of copy/paste in the following - I guess we can condense this
      // down a lot with a well chosen function call

      try {

         uint8_t bitpattern[config->length];
         std::cout<< "Printing config:"<<std::endl;
         for (unsigned int nbit = 0; nbit < config->length; ++nbit) {
             for(short i=0;i<8;i++){// reverse Bits (reverse config setting is something different !!)
                  bitpattern[nbit] |= ((config->bitpattern_w[nbit]>>i) & 0b1)<<(7-i);
             }
         }

         uint32_t * datastream = (uint32_t*)(bitpattern);

         vector<uint32_t> payload;
         for (unsigned int nbit = 0; nbit < config->length_32bits; ++nbit) {
             uint32_t tmp = ((datastream[nbit]>>24)&0x000000FF) | ((datastream[nbit]>>8)&0x0000FF00) | ((datastream[nbit]<<8)&0x00FF0000) | ((datastream[nbit]<<24)&0xFF000000);
             payload.push_back(tmp);
             std::cout << std::hex << tmp << std::endl;
         }

         // ToDo: Col Test Tdac bits from file
         for(int i=0; i<85;i++){
             payload.push_back(0x00000000);
         }
         std::cout<<"length 32:"<<config->length_32bits<<std::endl;
         std::cout<<"length byte:"<<config->length<<std::endl;

         //Mask all chips but this one
         uint32_t chip_select_mask = 0xfff; //all chips masked (12 times 1)
         int pos = ASICid_from_ID(asic);
         chip_select_mask &= ((~0x1) << pos);
         for (int i = 0; i < pos; ++i)
             chip_select_mask |= (0x1 << i);
         std::cout << "Chip select mask = " << std::hex << chip_select_mask << std::endl;
         feb_sc.FEB_write(SP_ID, 0xFF48, chip_select_mask);


         // TODO: include headers for addr.
         feb_sc.FEB_write(SP_ID, 0xFF47, 0x0000000F); // SPI slow down reg
         feb_sc.FEB_write(SP_ID, 0xFF40, 0x00000FC0); // reset Mupix config fifos
         feb_sc.FEB_write(SP_ID, 0xFF40, 0x00000000);
         feb_sc.FEB_write(SP_ID, 0xFF49, 0x00000003); // idk, have to look it up
         rpc_status = feb_sc.FEB_write(SP_ID, 0xFF4A, payload,true);

      } catch(std::exception& e) {
          cm_msg(MERROR, "setup_mupix", "Communication error while configuring MuPix %d: %s", asic, e.what());
          set_equipment_status(equipment_name, "SB-FEB Communication error", "red");
          return FE_ERR_HW; //note: return of lambda function
      }
      if(rpc_status!=FEB_REPLY_SUCCESS){
         //configuration mismatch, report and break foreach-loop
         set_equipment_status(equipment_name,  "MuPix config failed", "red");
         cm_msg(MERROR, "setup_mupix", "MuPix configuration error for ASIC %i", asic);
         return FE_ERR_HW;//note: return of lambda function
      }

      return FE_SUCCESS;//note: return of lambda function
   });//MapForEach
   return status; //status of foreach function, SUCCESS when no error.
}

//MIDAS callback function for FEB register Setter functions
void MupixFEB::on_settings_changed(odb o, void * userdata)
{
    std::string name = o.get_name();

    cm_msg(MINFO, "MupixFEB::on_settings_changed", "Setting changed (%s)", name.c_str());

    MupixFEB* _this=static_cast<MupixFEB*>(userdata);
    
    BOOL bval;

    if (name == "dummy_config") {
        bval = o;
        
        cm_msg(MINFO, "MupixFEB::on_settings_changed", "Set dummy_config %d", bval);
        // TODO: do something here
        //     _this->setDummyConfig(MupixFEB::FPGA_broadcast_ID,value);
   }

   if (name == "reset_asics") {
      bval = o;
      if(bval){
         cm_msg(MINFO, "MupixFEB::on_settings_changed", "reset_asics");
         // TODO: do something here
//         _this->chipReset(MupixFEB::FPGA_broadcast_ID);
         o = FALSE; // reset flag in ODB
      }
   }
   
   if (name == "reset_boards") {
      bval = o;
      if(bval){
         cm_msg(MINFO, "MupixFEB::on_settings_changed", "reset_boards");
         // TODO: do something here
//         _this->chipReset(MupixFEB::FPGA_broadcast_ID);
        o = FALSE; // reset flag in ODB
      }
   }

}

unsigned char reverse(unsigned char b) {
   b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
   b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
   b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
   return b;
}

// TODO: The following two functions do the same???
uint32_t MupixFEB::ReadBackCounters(uint16_t FPGA_ID){
   //map to SB fiber
   auto FEB = febs[FPGA_ID];
   if(!FEB.IsScEnabled()) return SUCCESS; //skip disabled fibers
   if(FEB.SB_Number()!= SB_number) return SUCCESS; //skip commands not for this SB

   vector<uint32_t> hitsEna(1);
   // TODO: Get rid of hardcoded address
    feb_sc.FEB_register_read(FEB.SB_Port(), 0x9a, hitsEna);
   return hitsEna[0];
}

uint32_t MupixFEB::ReadBackHitsEnaRate(uint16_t FPGA_ID){
    auto FEB = febs[FPGA_ID];
    if(!FEB.IsScEnabled()) return SUCCESS; //skip disabled fibers
    if(FEB.SB_Number()!= SB_number) return SUCCESS; //skip commands not for this SB
    
    vector<uint32_t> hitsEna(1);
    // TODO: Get rid of hardcoded address
    feb_sc.FEB_register_read(FEB.SB_Port(), 0x9a, hitsEna);
    return hitsEna[0];
}
