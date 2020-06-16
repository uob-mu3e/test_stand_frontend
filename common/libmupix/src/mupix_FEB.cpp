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

#include "mudaq_device_scifi.h"
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

MupixFEB* MupixFEB::m_instance=NULL;

//Mapping to physical ports of switching board.
uint16_t MupixFEB::FPGAid_from_ID(int asic){return asic/4;}
uint16_t MupixFEB::ASICid_from_ID(int asic){return asic%4;}

uint16_t MupixFEB::GetNumASICs(){return m_FPGAs.size()*4;} //TODO: add parameter for number of asics per FEB, later more flexibility to have different number of sensors per FEB

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

//ASIC configuration:
//Configure all asics under prefix (e.g. prefix="/Equipment/Mupix")
int MupixFEB::ConfigureASICs(){
   printf("MupixFEB::ConfigureASICs()\n");
   cm_msg(MINFO, "MupixFEB" , "Configuring sensors under prefix %s/Settings/ASICs/", m_odb_prefix);
   int status = mupix::midasODB::MapForEachASIC(hDB,m_odb_prefix,[this](mupix::MupixConfig* config, int asic){
      uint32_t rpc_status;
      //mapping
      uint16_t SB_ID=m_FPGAs[FPGAid_from_ID(asic)].SB_Number();
      uint16_t SP_ID=m_FPGAs[FPGAid_from_ID(asic)].SB_Port();
      uint16_t FA_ID=ASICid_from_ID(asic);

      if(!m_FPGAs[FPGAid_from_ID(asic)].IsScEnabled()){
          printf(" [skipped]\n");
          return FE_SUCCESS;
      }
      if(SB_ID!=m_SB_number){
          printf(" [skipped]\n");
          return FE_SUCCESS;
      } //TODO

      cm_msg(MINFO, "MupixFEB" , "Configuring sensor %s/Settings/ASICs/%i/: Mapped to FEB%u -> SB%u.%u  ASIC #%d", m_odb_prefix,asic,FPGAid_from_ID(asic),SB_ID,SP_ID,FA_ID);


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
         rpc_status=m_mu.FEBsc_NiosRPC(SP_ID,0x0110,{{reinterpret_cast<uint32_t*>(&asic),1},{reinterpret_cast<uint32_t*>(datastream), config->length_32bits}});

      } catch(std::exception& e) {
          cm_msg(MERROR, "setup_mupix", "Communication error while configuring MuPix %d: %s", asic, e.what());
          set_equipment_status(m_equipment_name, "SB-FEB Communication error", "red");
          return FE_ERR_HW; //note: return of lambda function
      }
      if(rpc_status!=FEB_REPLY_SUCCESS){
         //configuration mismatch, report and break foreach-loop
         set_equipment_status(m_equipment_name,  "MuPix config failed", "red");
         cm_msg(MERROR, "setup_mupix", "MuPix configuration error for ASIC %i", asic);
         return FE_ERR_HW;//note: return of lambda function
      }

      for (int rrow = 0; rrow < 200; ++rrow) {
          try {
             uint32_t * datastream = (uint32_t*)(default_config_mupix[rrow]);

             for (unsigned int nbit = 0; nbit < config->length_32bits; ++nbit) {
                 uint32_t tmp = ((datastream[nbit]>>24)&0x000000FF) | ((datastream[nbit]>>8)&0x0000FF00) | ((datastream[nbit]<<8)&0x00FF0000) | ((datastream[nbit]<<24)&0xFF000000);\
                 datastream[nbit] = tmp;
             }
             rpc_status=m_mu.FEBsc_NiosRPC(SP_ID,0x0110,{{reinterpret_cast<uint32_t*>(&asic),1},{reinterpret_cast<uint32_t*>(datastream), config->length_32bits}});


          } catch(std::exception& e) {
              cm_msg(MERROR, "setup_mupix", "Communication error while configuring MuPix %d: %s", asic, e.what());
              set_equipment_status(m_equipment_name, "SB-FEB Communication error", "red");
              return FE_ERR_HW; //note: return of lambda function
          }
          if(rpc_status!=FEB_REPLY_SUCCESS){
             //configuration mismatch, report and break foreach-loop
             set_equipment_status(m_equipment_name,  "MuPix config failed", "red");
             cm_msg(MERROR, "setup_mupix", "MuPix configuration error for ASIC %i", asic);
             return FE_ERR_HW;//note: return of lambda function
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
         rpc_status=m_mu.FEBsc_NiosRPC(SP_ID,0x0110,{{reinterpret_cast<uint32_t*>(&asic),1},{reinterpret_cast<uint32_t*>(datastream), config->length_32bits}});

      } catch(std::exception& e) {
          cm_msg(MERROR, "setup_mupix", "Communication error while configuring MuPix %d: %s", asic, e.what());
          set_equipment_status(m_equipment_name, "SB-FEB Communication error", "red");
          return FE_ERR_HW; //note: return of lambda function
      }
      if(rpc_status!=FEB_REPLY_SUCCESS){
         //configuration mismatch, report and break foreach-loop
         set_equipment_status(m_equipment_name,  "MuPix config failed", "red");
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
    
    INT ival;
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


int MupixFEB::ConfigureBoards(){
   cm_msg(MINFO, "MupixFEB" , "Configuring boards under prefix %s/Settings/Boards/", m_odb_prefix);
   int status = mupix::midasODB::MapForEachBOARD(hDB,m_odb_prefix,[this](mupix::MupixBoardConfig* config, int board){
      uint32_t rpc_status;
      //mapping
      uint16_t SB_ID=m_FPGAs[FPGAid_from_ID(board)].SB_Number();
      uint16_t SP_ID=m_FPGAs[FPGAid_from_ID(board)].SB_Port();

      uint32_t board_ID=board;

      if(!m_FPGAs[FPGAid_from_ID(board)].IsScEnabled()){
          printf(" [skipped]\n");
          return FE_SUCCESS;
      }
      if(SB_ID!=m_SB_number){
          printf(" [skipped]\n");
          return FE_SUCCESS;
      } //TODO
      //printf("\n");

      cm_msg(MINFO, "MupixFEB" , "Configuring MuPIX board %s/Settings/Boards/%i/: Mapped to FEB%u -> SB%u.%u", m_odb_prefix,board,FPGAid_from_ID(board),SB_ID,SP_ID);

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
           rpc_status=m_mu.FEBsc_NiosRPC(SP_ID,0x0120,{{reinterpret_cast<uint32_t*>(&board),1},{reinterpret_cast<uint32_t*> (datastream), config->length_32bits}});

      } catch(std::exception& e) {
          cm_msg(MERROR, "setup_mupix", "Communication error while configuring MuPix %d: %s", board, e.what());
          set_equipment_status(m_equipment_name, "SB-FEB Communication error", "red");
          return FE_ERR_HW; //note: return of lambda function
      }
      if(rpc_status!=FEB_REPLY_SUCCESS){
         //configuration mismatch, report and break foreach-loop
         set_equipment_status(m_equipment_name,  "MuPix config failed", "red");
         cm_msg(MERROR, "setup_mupix", "MuPix configuration error for Board %i", board);
      }

      return FE_SUCCESS;//note: return of lambda function
   });//MapForEach
   return status; //status of foreach function, SUCCESS when no error.
}


//Helper functions
uint32_t reg_setBit  (uint32_t reg_in, uint8_t bit, bool value=true){
	if(value)
		return (reg_in | 1<<bit);
	else
		return (reg_in & (~(1<<bit)));
}
uint32_t reg_unsetBit(uint32_t reg_in, uint8_t bit){return reg_setBit(reg_in,bit,false);}

bool reg_getBit(uint32_t reg_in, uint8_t bit){
	return (reg_in & (1<<bit)) != 0;
}

uint32_t reg_getRange(uint32_t reg_in, uint8_t length, uint8_t offset){
	return (reg_in>>offset) & ((1<<length)-1);
}
uint32_t reg_setRange(uint32_t reg_in, uint8_t length, uint8_t offset, uint32_t value){
	return (reg_in & ~(((1<<length)-1)<<offset)) | ((value & ((1<<length)-1))<<offset);
}


//MupixFEB registers and functions
/*
void MupixFEB::setDummyConfig(int FPGA_ID, bool dummy){
	printf("MupixFEB::setDummyConfig(%d)=%d\n",FPGA_ID,dummy);
	uint32_t val;

        //TODO: shadowing should know about broadcast FPGA ID
	//TODO: implement pull from FPGA when shadow value is not stored
	//m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
        //printf("MupixFEB(%d)::FE_DUMMYCTRL_REG readback=%8.8x\n",FPGA_ID,val);
        val=m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG];

        val=reg_setBit(val,0,dummy);
	printf("MupixFEB(%d)::FE_DUMMYCTRL_REG new=%8.8x\n",FPGA_ID,val);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG,false);
	m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG]=val;
}
*/
