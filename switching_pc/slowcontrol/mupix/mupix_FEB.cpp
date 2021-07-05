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

for ( int asic=0; asic<10; asic++ ) {
    
    if ( asic == 0 ) {
        //feb_sc.FEB_register_write(0, MP_LVDS_LINK_MASK_REGISTER_W, 0x1C7);
        //feb_sc.FEB_register_write(0, MP_LVDS_LINK_MASK2_REGISTER_W, 0x7);
        feb_sc.FEB_register_write(0, MP_LVDS_LINK_MASK_REGISTER_W, 0x004C01AD);
        feb_sc.FEB_register_write(0, MP_LVDS_LINK_MASK2_REGISTER_W, 0x0);
    }
                                                                                
    if ( asic == 1 ) {
        //feb_sc.FEB_register_write(1, MP_LVDS_LINK_MASK2_REGISTER_W, 0xE);
        feb_sc.FEB_register_write(1, MP_LVDS_LINK_MASK_REGISTER_W, 0xA000002C);
        feb_sc.FEB_register_write(1, MP_LVDS_LINK_MASK2_REGISTER_W, 0x00000002);
    }
                                                                                
    if ( asic == 2 ) {
        //feb_sc.FEB_register_write(2, MP_LVDS_LINK_MASK_REGISTER_W, 0xE);
        feb_sc.FEB_register_write(2, MP_LVDS_LINK_MASK_REGISTER_W, 0xF8008A2C);
        feb_sc.FEB_register_write(2, MP_LVDS_LINK_MASK2_REGISTER_W, 0x0000000F);
    }
                                                                                
    if ( asic == 3 ) {
        //feb_sc.FEB_register_write(3, MP_LVDS_LINK_MASK_REGISTER_W, 0xC003F03F);
        feb_sc.FEB_register_write(3, MP_LVDS_LINK_MASK_REGISTER_W, 0x0);
        //feb_sc.FEB_register_write(3, MP_LVDS_LINK_MASK2_REGISTER_W, 0x1);
        feb_sc.FEB_register_write(3, MP_LVDS_LINK_MASK2_REGISTER_W, 0x0);
    }
                                                                                
    if ( asic == 4 ) {
        feb_sc.FEB_register_write(4, MP_LVDS_LINK_MASK_REGISTER_W, 0xFFFD5000);
        feb_sc.FEB_register_write(4, MP_LVDS_LINK_MASK2_REGISTER_W, 0x0000000F);
    }
                                                                                
    if ( asic == 5 ) {
        //feb_sc.FEB_register_write(5, MP_LVDS_LINK_MASK_REGISTER_W, 0x38000);
        feb_sc.FEB_register_write(5, MP_LVDS_LINK_MASK_REGISTER_W, 0xA0008A00);
        //feb_sc.FEB_register_write(5, MP_LVDS_LINK_MASK2_REGISTER_W, 0xE);
        feb_sc.FEB_register_write(5, MP_LVDS_LINK_MASK2_REGISTER_W, 0x00000002);
    }
                                                                              
    if ( asic == 6 ) {
        //feb_sc.FEB_register_write(6, MP_LVDS_LINK_MASK_REGISTER_W, 0x381FF);
        feb_sc.FEB_register_write(6, MP_LVDS_LINK_MASK_REGISTER_W, 0x07FC8BFF);
        feb_sc.FEB_register_write(6, MP_LVDS_LINK_MASK2_REGISTER_W, 0x0);
    }
                                                                               
    if ( asic == 7 ) {
        //feb_sc.FEB_register_write(7, MP_LVDS_LINK_MASK_REGISTER_W, 0x1C0);
        feb_sc.FEB_register_write(7, MP_LVDS_LINK_MASK_REGISTER_W, 0x0000002C);
        feb_sc.FEB_register_write(7, MP_LVDS_LINK_MASK2_REGISTER_W, 0x0);
    }
                                                                               
    if ( asic == 8 ) {
        //feb_sc.FEB_register_write(8, MP_LVDS_LINK_MASK_REGISTER_W, 0x7);
        feb_sc.FEB_register_write(8, MP_LVDS_LINK_MASK_REGISTER_W, 0xF80001FF);
        feb_sc.FEB_register_write(8, MP_LVDS_LINK_MASK2_REGISTER_W, 0x0000000F);
    }

    if ( asic == 9 ) {
        //feb_sc.FEB_register_write(9, MP_LVDS_LINK_MASK_REGISTER_W, 0x7);
        feb_sc.FEB_register_write(9, MP_LVDS_LINK_MASK_REGISTER_W, 0xF8000000);
        feb_sc.FEB_register_write(9, MP_LVDS_LINK_MASK2_REGISTER_W, 0x0000000F);
    }
}

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
          uint32_t bitpattern_m;
          vector<vector<uint32_t> > payload_m;
          vector<uint32_t> payload;
          payload_m.push_back(vector<uint32_t>(reinterpret_cast<uint32_t*>(config->bitpattern_w),reinterpret_cast<uint32_t*>(config->bitpattern_w)+config->length_32bits));

          for(uint32_t j = 0; j<payload_m.at(0).size();j++){
              bitpattern_m=0;
              for(short i=0; i<32; i++){
                  bitpattern_m|= ((payload_m.at(0).at(j)>>i) & 0b1)<<(31-i);
              }
              payload.push_back(bitpattern_m);
          }

          for(uint32_t j = 0; j<payload.size();j++){
              std::cout<<std::hex<<payload.at(j)<<std::endl;
          }

         // ToDo: Col Test Tdac bits from file
         for(int i=0; i<85;i++){
             payload.push_back(0x00000000);
         }

         //Mask all chips but this one
         uint32_t chip_select_mask = 0xfff; //all chips masked (12 times 1)
         int pos = ASICid_from_ID(asic);
         chip_select_mask &= ((~0x1) << pos);
         for (int i = 0; i < pos; ++i)
             chip_select_mask |= (0x1 << i);

         uint32_t spi_busy = 1;
         uint32_t count = 0;
         uint32_t limit = 5;
         rpc_status=FEB_REPLY_SUCCESS;

         feb_sc.FEB_register_read(SP_ID,0x4B,spi_busy);

         while(spi_busy==1 && count < limit){
             sleep(1);
             feb_sc.FEB_register_read(SP_ID,0x4B,spi_busy);
             count++;
             cm_msg(MINFO, "MupixFEB", "Mupix config spi busy .. waiting");
         }
         if(count == limit){
             std::cout<<"Timeout"<<std::endl;
             cm_msg(MERROR, "setup_mupix", "FEB Mupix SPI timeout");
         }else{
            
            feb_sc.FEB_write(SP_ID, 0xFF48, chip_select_mask);

            // TODO: include headers for addr.
            feb_sc.FEB_write(SP_ID, 0xFF47, 0x0000000F); // SPI slow down reg
            feb_sc.FEB_write(SP_ID, 0xFF40, 0x00000FC0); // reset Mupix config fifos
            feb_sc.FEB_write(SP_ID, 0xFF40, 0x00000000);
            feb_sc.FEB_write(SP_ID, 0xFF49, 0x00000003); // idk, have to look it up
	    
            // We now only write the default configuration for testing
            rpc_status = feb_sc.FEB_write(SP_ID, 0xFF4A, payload,true);
           
         }
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

      // reset lvds links
      feb_sc.FEB_register_write(SP_ID, MP_RESET_LVDS_N_REGISTER_W, 0x0);
      feb_sc.FEB_register_write(SP_ID, MP_RESET_LVDS_N_REGISTER_W, 0x1);

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

uint32_t MupixFEB::ReadBackLVDSStatus(DWORD* pdata, uint16_t FPGA_ID, uint16_t LVDS_ID)
{
    auto FEB = febs.at(FPGA_ID);

    //skip disabled fibers
    if(!FEB.IsScEnabled())
        return 0;

    //skip commands not for this SB
    if(FEB.SB_Number()!=SB_number)
        return 0;
    
    uint32_t val;
    int status = feb_sc.FEB_register_read(FEB.SB_Port(), MP_LVDS_STATUS_START_REGISTER_W + LVDS_ID, val);
    
    return val;
}

uint32_t MupixFEB::ReadBackLVDSNumHits(uint16_t FPGA_ID, uint16_t LVDS_ID)
{
    //cm_msg(MINFO, "MupixFEB::ReadBackLVDSNumHits" , "Implement Me");
    return 0;
}

uint32_t MupixFEB::ReadBackLVDSNumHitsInMupixFormat(uint16_t FPGA_ID, uint16_t LVDS_ID)
{
    //cm_msg(MINFO, "MupixFEB::ReadBackLVDSNumHitsInMupixFormat" , "Implement Me");
    return 0;
}

DWORD* MupixFEB::ReadLVDSCounters(DWORD* pdata, uint16_t FPGA_ID)
{
    for(uint32_t i=0; i<lvds_links_per_feb; i++){ // TODO: set currect LVDS links number
        // Link ID
        *(DWORD*)pdata++ = i;
        // read lvds status
        *(DWORD*)pdata++ = ReadBackLVDSStatus(pdata, FPGA_ID, i);
        // number of hits from link
        *(DWORD*)pdata++ = ReadBackLVDSNumHits(FPGA_ID, i);
        // number of hits from link in mupix format
        *(DWORD*)pdata++ = ReadBackLVDSNumHitsInMupixFormat(FPGA_ID, i);

    };
    return pdata;
}
