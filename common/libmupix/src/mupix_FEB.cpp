/********************************************************************\

  Name:         Mupix_FEB.h
  Created by:   Konrad Briggl

Contents:       Definition of functions to talk to a mupix-based FEB. Designed to no be derived into a SciFi_FEB and a Tiles_FEB class where subdetector specific actions are defined.
		Here: Definition of basic things for mupix-configuration & datapath settings

\********************************************************************/

#include "mupix_FEB.h"
#include "midas.h"
#include "mfe.h" //for set_equipment_status

#include "mudaq_device_scifi.h"
#include "mupix_config.h"
#include "mupix_midasodb.h"
#include <thread>
#include <chrono>

//offset for registers on nios SC memory
#define SC_REG_OFFSET 0xff60
#define FE_DUMMYCTRL_REG       (SC_REG_OFFSET+0x8)
#define FE_DPCTRL_REG          (SC_REG_OFFSET+0x9)
#define FE_SUBDET_RESET_REG    (SC_REG_OFFSET+0xa)
#define FE_SPIDATA_ADDR		0

const uint8_t MupixFEB::FPGA_broadcast_ID=0;

MupixFEB* MupixFEB::m_instance=NULL;

//Mapping to physical ports of switching board.
uint8_t MupixFEB::FPGAid_from_ID(int asic){return 0;}//return asic/4;}
uint8_t MupixFEB::ASICid_from_ID(int asic){return asic;}//return asic%4;}

//ASIC configuration:
//Configure all asics under prefix (e.g. prefix="/Equipment/Mupix")
int MupixFEB::ConfigureASICs(HNDLE hDB, const char* equipment_name, const char* odb_prefix){
   printf("MupixFEB::ConfigureASICs()\n");
   int status = mupix::midasODB::MapForEachASIC(hDB,odb_prefix,[this,&odb_prefix,&equipment_name](mupix::MupixConfig* config, int asic){
      int status=SUCCESS;
      uint32_t reg;
      cm_msg(MINFO, "setup_mupix" , "Configuring MuPIX asic %s/Settings/ASICs/%i/", odb_prefix, asic);

      try {
         //Write ASIC number
         reg=asic;
         m_mu.FEBsc_write(FPGAid_from_ID(asic), &reg, 1, (uint32_t) FE_SPIDATA_ADDR,true);
	 printf("reading back\n");
         m_mu.FEBsc_read(FPGAid_from_ID(asic), &reg, 1,  (uint32_t) FE_SPIDATA_ADDR,true);
         //Write configuration
         m_mu.FEBsc_write(FPGAid_from_ID(asic), reinterpret_cast<uint32_t*>(config->bitpattern_w), config->length_32bits , (uint32_t) FE_SPIDATA_ADDR+1,true);

         //Write offset address
         reg= FE_SPIDATA_ADDR;
         m_mu.FEBsc_write(FPGAid_from_ID(asic), &reg,1,0xfff1,true);

         //Write command word to register FFF0: cmd | n
         reg= 0x01100000 + (0xFFFF & config->length_32bits);
         m_mu.FEBsc_write(FPGAid_from_ID(asic), &reg,1,0xfff0,true);

         //Wait for configuration to finish
         uint timeout_cnt = 0;
         do{
            printf("Polling (%d)\n",timeout_cnt);
            if(++timeout_cnt >= 10000) throw std::runtime_error("SPI transaction timeout while configuring asic"+std::to_string(asic));
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            m_mu.FEBsc_read(FPGAid_from_ID(asic), &reg, 1, 0xfff0);
         }while( (reg&0xffff0000) != 0);
      } catch(std::exception& e) {
          cm_msg(MERROR, "setup_mupix", "Communication error while configuring MuTRiG %d: %s", asic, e.what());
          set_equipment_status(equipment_name, "SB-FEB Communication error", "red");
          return FE_ERR_HW; //note: return of lambda function
      }
      if(status!=SUCCESS){
         //configuration mismatch, report and break foreach-loop
         set_equipment_status(equipment_name,  "MuTRiG config failed", "red");
         cm_msg(MERROR, "setup_mupix", "MuTRiG configuration error for ASIC %i", asic);
      }

      return status;//note: return of lambda function
   });//MapForEach
   return status; //status of foreach function, SUCCESS when no error.
}

//MIDAS callback function for FEB register Setter functions
void MupixFEB::on_settings_changed(HNDLE hDB, HNDLE hKey, INT, void * userdata)
{
   MupixFEB* _this=static_cast<MupixFEB*>(userdata);
   KEY key;
   db_get_key(hDB, hKey, &key);
   printf("MupixFEB::on_settings_changed(%s)\n",key.name);
   if (std::string(key.name) == "dummy_config") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      cm_msg(MINFO, "MupixFEB::on_settings_changed", "Set dummy_config to %d", value);
 //     _this->setDummyConfig(MupixFEB::FPGA_broadcast_ID,value);
   }
   if (std::string(key.name) == "reset_asics") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      if(value){
         cm_msg(MINFO, "MupixFEB::on_settings_changed", "reset_asics");
//         _this->chipReset(MupixFEB::FPGA_broadcast_ID);
         value = FALSE; // reset flag in ODB
         db_set_data(hDB, hKey, &value, sizeof(value), 1, TID_BOOL);
      }
   }
   
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
