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
uint16_t MupixFEB::FPGAid_from_ID(int asic) const {return asic/2;}
uint16_t MupixFEB::ASICid_from_ID(int asic) const {return asic%2;}

uint16_t MupixFEB::GetNumASICs() const {return febs.size()*2;} //TODO: add parameter for number of asics per FEB, later more flexibility to have different number of sensors per FEB

uint32_t default_mupix_dacs[94] =
{
0x12d3e2,
0x1abc52d0,
0x0,
0x30000000,
0x0,
0x0,
0x0,
0xcc000,
0x0,
0x30c880c2,
0x80000000,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x0,
0x9e100000,
0xe,
0x80ffff,
0x608618,
0x6145186d,
0x24000030,
0xc0200,
0xa
};

void invert_datastream(uint32_t * datastream) {

}

u_int32_t transform_row_dac(u_int32_t row)    //physical row to ram address (For dac WRITE ONLY, read transformation is different again !!)
{
    u_int32_t newrow;

    if(row<85)
        newrow = 199-row;
    else if(row>99)
        newrow = 215-row;
    else
        newrow = 99-row;

    return newrow;
}

short transform_tdac(short tDAC){
    // this is a mess ..
    // bit(0)=1 is mask pixel
    // bit(1::2): comparator 1, bit(3::5) comparator 2, 6 inject, 7 hitbus seems NOT to work !!! (to be tested again)
    // TODO: figure out the correct bit order

    // for now we just turn the pixel off if there is something
    if(tDAC!=0)
        return 1;
    return 0;
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

         uint8_t bitpatterna[config->length +1];
         for (unsigned int nbit = 0; nbit < config->length; ++nbit) {
             bitpatterna[nbit+1] = config->bitpattern_w[nbit];
         }
         uint32_t * datastream = (uint32_t*)(bitpatterna);

         for (unsigned int nbit = 0; nbit < config->length_32bits; ++nbit) {
             uint32_t tmp = ((datastream[nbit]>>24)&0x000000FF) | ((datastream[nbit]>>8)&0x0000FF00) | ((datastream[nbit]<<8)&0x00FF0000) | ((datastream[nbit]<<24)&0xFF000000);\
             datastream[nbit] = tmp;
         }
         vector<vector<uint32_t> > payload;
         payload.push_back(vector<uint32_t>(1,asic));
         payload.push_back(vector<uint32_t>(reinterpret_cast<uint32_t*>(datastream),reinterpret_cast<uint32_t*>(datastream)+config->length_32bits));
         rpc_status = feb_sc.FEBsc_NiosRPC(SP_ID, feb::CMD_MUPIX_CHIP_CFG, payload);

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

      midas::odb mpTDACs("/Equipment/MuPix/TDACs");
      midas::odb mpColDACs("/Equipment/MuPix/Settings/Coldacs");
      //midas::odb mpRowDACs("/Equipment/MuPix/Settings/DigitalRowdacs");
      midas::odb mpRowDACs("/Equipment/MuPix/Settings/DigitalRowdacs");
      useTDACs=mpTDACs["useTDACs"];

      if(useTDACs==1){
         // use TDAC config files
         mpTDACs["chipIDreq"]=asic;  // request load of TDACs of this chip ID
         for(int i=0; i<5;i++){
            sleep(1);               // TODO: preload TDACs and save locally to avoid this (We are waiting for the TDAC loading from (remote) disk into odb here)
            if(mpTDACs["chipIDactual"]==asic){
                cm_msg(MINFO, "setup_mupix", "loading TDACs for ASIC %i", asic);

                for (int rrow = 0; rrow < 200; rrow++) {
                    // rrow: this is the real row of the mupix
                    // rowRAM_addr : address of this row (For dac WRITE ONLY, read transformation is different again !!)
                    rowRAM_addr=transform_row_dac((u_int32_t) rrow);

                    try {
                       mpRowDACs[(std::to_string(asic)+"/row_"+std::to_string(rowRAM_addr)+"/digiWrite").c_str()]=1;

                       for(int col=0;col<127;col++){
                           tDAC=mpTDACs[("col"+std::to_string(col)).c_str()][rrow];
                           tDAC=transform_tdac(tDAC);
                           mpColDACs[(std::to_string(asic)+"/col_"+std::to_string(col)+"/RAM").c_str()]=tDAC;
                       }

                       uint8_t bitpatterna[config->length +1];
                       for (unsigned int nbit = 0; nbit < config->length; ++nbit) {
                           bitpatterna[nbit+1] = config->bitpattern_w[nbit];
                       }
                       uint32_t * datastream = (uint32_t*)(bitpatterna);

                       mpRowDACs[(std::to_string(asic)+"/row_"+std::to_string(rowRAM_addr)+"/digiWrite").c_str()]=0;

                       for (unsigned int nbit = 0; nbit < config->length_32bits; ++nbit) {
                           uint32_t tmp = ((datastream[nbit]>>24)&0x000000FF) | ((datastream[nbit]>>8)&0x0000FF00) | ((datastream[nbit]<<8)&0x00FF0000) | ((datastream[nbit]<<24)&0xFF000000);\
                           datastream[nbit] = tmp;
                       }
                       vector<vector<uint32_t> > payload;
                       payload.push_back(vector<uint32_t>(1,asic));
                       payload.push_back(vector<uint32_t>(reinterpret_cast<uint32_t*>(datastream),reinterpret_cast<uint32_t*>(datastream)+config->length_32bits));
                       // TODO: Get rid of hardcoded address!
                       rpc_status = feb_sc.FEBsc_NiosRPC(SP_ID, 0x0110, payload);

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
                }

                break;
            }
            if(i==4) {
                TDACsNotFound=true;
                cm_msg(MERROR, "setup_mupix", "failed to load TDACs for ASIC %i, continue with defaults ", asic);
                printf("not using TDACS");
            }
         }
      }

      if(useTDACs==0 || TDACsNotFound){
          // write 0's for all TDACs
          for (int rrow = 0; rrow < 200; ++rrow) {
              try {
                 uint32_t * datastream = (uint32_t*)(default_config_mupix[rrow]);

                 for (unsigned int nbit = 0; nbit < config->length_32bits; ++nbit) {
                     uint32_t tmp = ((datastream[nbit]>>24)&0x000000FF) | ((datastream[nbit]>>8)&0x0000FF00) | ((datastream[nbit]<<8)&0x00FF0000) | ((datastream[nbit]<<24)&0xFF000000);\
                     datastream[nbit] = tmp;
                 }
                 vector<vector<uint32_t> > payload;
                 payload.push_back(vector<uint32_t>(1,asic));
                 payload.push_back(vector<uint32_t>(reinterpret_cast<uint32_t*>(datastream),reinterpret_cast<uint32_t*>(datastream)+config->length_32bits));
                 // TODO: Get rid of hardcoded address!
                 rpc_status = feb_sc.FEBsc_NiosRPC(SP_ID, 0x0110, payload);

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
          }
      }

      try {
         uint8_t bitpatterna[config->length +1];
         for (unsigned int nbit = 0; nbit < config->length; ++nbit) {
             bitpatterna[nbit+1] = config->bitpattern_w[nbit];
         }
         uint32_t * datastream = (uint32_t*)(bitpatterna);

         for (unsigned int nbit = 0; nbit < config->length_32bits; ++nbit) {
             uint32_t tmp = ((datastream[nbit]>>24)&0x000000FF) | ((datastream[nbit]>>8)&0x0000FF00) | ((datastream[nbit]<<8)&0x00FF0000) | ((datastream[nbit]<<24)&0xFF000000);\
             datastream[nbit] = tmp;
         }
         vector<vector<uint32_t> > payload;
         payload.push_back(vector<uint32_t>(1,asic));
         payload.push_back(vector<uint32_t>(reinterpret_cast<uint32_t*>(datastream),reinterpret_cast<uint32_t*>(datastream)+config->length_32bits));
         // TODO: Get rid of hardcoded address!
         rpc_status = feb_sc.FEBsc_NiosRPC(SP_ID, 0x0110, payload);

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

int MupixFEB::ConfigureBoards(){
   cm_msg(MINFO, "MupixFEB" , "Configuring boards under prefix %s/Settings/Boards/", odb_prefix);
   int status = mupix::midasODB::MapForEachBOARD(hDB,odb_prefix,[this](mupix::MupixBoardConfig* config, int board){
      uint32_t rpc_status;
      //mapping
      uint16_t SB_ID=febs[FPGAid_from_ID(board)].SB_Number();
      uint16_t SP_ID=febs[FPGAid_from_ID(board)].SB_Port();

      uint32_t board_ID=board;

      if(!febs[FPGAid_from_ID(board)].IsScEnabled()){
          printf(" [skipped]\n");
          return FE_SUCCESS;
      }
      if(SB_ID!= SB_number){
          printf(" [skipped]\n");
          return FE_SUCCESS;
      } //TODO
      //printf("\n");

      cm_msg(MINFO, "MupixFEB" ,
             "Configuring MuPIX board %s/Settings/Boards/%i/: Mapped to FEB%u -> SB%u.%u",
             odb_prefix,board,FPGAid_from_ID(board),SB_ID,SP_ID);

       uint8_t bitpattern[config->length +1];
       for (unsigned int nbit = 0; nbit < config->length; ++nbit) {
           bitpattern[nbit] = (uint8_t) reverse((unsigned char) config->bitpattern_w[nbit]);
       }

       uint32_t * datastream = (uint32_t*)(bitpattern);

       for (unsigned int nbit = 0; nbit < config->length_32bits; ++nbit) {
           uint32_t tmp = ((datastream[nbit]>>8)&0x00FF0000) | ((datastream[nbit]<<8)&0xFF000000) | ((datastream[nbit]>>8)&0x000000FF) | ((datastream[nbit]<<8)&0x0000FF00);
           datastream[nbit] = tmp;
       }
       try {
           vector<vector<uint32_t> > payload;
           payload.push_back(vector<uint32_t>(1,board));
           payload.push_back(vector<uint32_t>(reinterpret_cast<uint32_t*>(datastream),reinterpret_cast<uint32_t*>(datastream)+config->length_32bits));
           // TODO: Get rid of hardcoded address!
           rpc_status = feb_sc.FEBsc_NiosRPC(SP_ID, feb::CMD_MUPIX_BOARD_CFG, payload);
      } catch(std::exception& e) {
          cm_msg(MERROR, "setup_mupix", "Communication error while configuring MuPix %d: %s", board, e.what());
          set_equipment_status(equipment_name, "SB-FEB Communication error", "red");
          return FE_ERR_HW; //note: return of lambda function
      }
      if(rpc_status!=FEB_REPLY_SUCCESS){
         //configuration mismatch, report and break foreach-loop
         set_equipment_status(equipment_name,  "MuPix config failed", "red");
         cm_msg(MERROR, "setup_mupix", "MuPix configuration error for Board %i", board);
      }

      return FE_SUCCESS;//note: return of lambda function
   });//MapForEach
   return status; //status of foreach function, SUCCESS when no error.
}
