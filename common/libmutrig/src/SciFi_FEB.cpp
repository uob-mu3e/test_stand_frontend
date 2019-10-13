/********************************************************************\

  Name:         FEB_access.h
  Created by:   Konrad Briggl

Contents:       Definition of fumctions in namespace mutrig
		to provide an abstraction layer to the (slow control) functions on the FE-FPGA

\********************************************************************/

#include "SciFi_FEB.h"
#include "midas.h"
#include "mfe.h" //for set_equipment_status

#include "mudaq_device_scifi.h"
#include "mutrig_config.h"
#include "mutrig_midasodb.h"
#include <thread>
#include <chrono>

//offset for registers on nios SC memory
#define SC_REG_OFFSET 0xff60
#define FE_DUMMYCTRL_REG       (SC_REG_OFFSET+0x8)
#define FE_DPCTRL_REG          (SC_REG_OFFSET+0x9)
#define FE_SUBDET_RESET_REG    (SC_REG_OFFSET+0xa)


SciFiFEB* SciFiFEB::m_instance=NULL;
const uint8_t SciFiFEB::FPGA_broadcast_ID=0;

//Mapping to physical ports. TODO: should be more configurable later, even from ODB?
uint8_t SciFiFEB::FPGAid_from_ID(int asic){return asic/4;}
uint8_t SciFiFEB::ASICid_from_ID(int asic){return asic%4;}

//ASIC configuration:
//Configure all asics under prefix (e.g. prefix="/Equipment/SciFi")
int SciFiFEB::ConfigureASICs(HNDLE hDB, const char* equipment_name, const char* odb_prefix){
   printf("SciFiFEB::ConfigureASICs()\n");
/*
   int status = mutrig::midasODB::MapForEach(hDB,odb_prefix,[this,&odb_prefix,&equipment_name](mutrig::Config* config, int asic){
   int status=SUCCESS;
   //try each asic twice, i.e. give up when cnt>1

   int cnt = 0;
   while(cnt<2) {
      try {
         cm_msg(MINFO, "setup_mutrig" , "Configuring MuTRiG asic %s/Settings/ASICs/%i/", odb_prefix, asic);
	 //Write configuration
//	 for(int i=0;i<10;i++) printf("pattern[%d]=%8.8x\n",i,config->bitpattern_w[i]);
	 m_mu.FEBsc_write(FPGAid_from_ID(asic), reinterpret_cast<uint32_t*>(config->bitpattern_w), config->length_32bits , (uint32_t) FE_SPIDATA_ADDR);
	 //Write handleID and start bit to trigger SPI transaction
	 uint32_t data=0;
	 data=SET_FE_SPICTRL_BIT_START(data);
	 data=SET_FE_SPICTRL_CHIPID_RANGE(data,ASICid_from_ID(asic));
	 m_mu.FEBsc_write(FPGAid_from_ID(asic), &data, 1, (uint32_t) FE_SPICTRL_REGISTER);
         //Wait for transaction to finish
         uint timeout_cnt = 0;
	 do{
            if(++timeout_cnt >= 10000) throw std::runtime_error("SPI transaction timeout while configuring asic"+std::to_string(asic));
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
	    m_mu.FEBsc_read(FPGAid_from_ID(asic), &data, 1, (uint32_t) FE_SPICTRL_REGISTER);

         }while(GET_FE_SPICTRL_BIT_START(data));
	 //Read back configuration
	 m_mu.FEBsc_read(FPGAid_from_ID(asic), reinterpret_cast<uint32_t*>(config->bitpattern_r), config->length_32bits , (uint32_t) FE_SPIDATA_ADDR);

	 status=config->VerifyReadbackPattern();
         if(status==SUCCESS) break; //configuration good, stopping here. Otherwise try another time without complaining here.
      } catch(std::exception& e) {
         cm_msg(MERROR, "setup_mutrig", "Communication error while configuring MuTRiG %d, try %d: %s", asic,cnt, e.what());
         set_equipment_status(equipment_name, "Communication error while configuring MuTRiG", "red");
         return FE_ERR_HW; //note: return of lambda function
      }
      cnt++;
   }
printf("Config class:\n");
   std::cout<<*config;
   if(status!=SUCCESS){
      //configuration mismatch, report and break foreach-loop
      set_equipment_status(equipment_name,  "MuTRiG config failed", "red");
      cm_msg(MERROR, "setup_mutrig", "MuTRiG configuration error for ASIC %i at try %d", asic, cnt);
      cm_msg(MERROR, "setup_mutrig", "%s",config->GetVerificationError().c_str());
      printf("Config class patterns after error condition seen:\n");
      std::cout<<*config;
   }
   return status;//note: return of lambda function
   });//MapForEach
   return status; //status of foreach function, SUCCESS when no error.
   */
   return 0;
}

//MIDAS callback function for FEB register Setter functions
void SciFiFEB::on_settings_changed(HNDLE hDB, HNDLE hKey, INT, void * userdata)
{
   SciFiFEB* _this=static_cast<SciFiFEB*>(userdata);
   KEY key;
   db_get_key(hDB, hKey, &key);
   printf("SciFiFEB::on_settings_changed(%s)\n",key.name);
   if (std::string(key.name) == "dummy_config") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      cm_msg(MINFO, "SciFiFEB::on_settings_changed", "Set dummy_config to %d", value);
      _this->setDummyConfig(SciFiFEB::FPGA_broadcast_ID,value);
   }
   if (std::string(key.name) == "dummy_data") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      cm_msg(MINFO, "SciFiFEB::on_settings_changed", "Set dummy_data to %d", value);
      _this->setDummyData_Enable(SciFiFEB::FPGA_broadcast_ID,value);
   }
   if (std::string(key.name) == "dummy_data_fast") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      cm_msg(MINFO, "SciFiFEB::on_settings_changed", "Set dummy_data_fast to %d", value);
      _this->setDummyData_Fast(SciFiFEB::FPGA_broadcast_ID,value);
   }
   if (std::string(key.name) == "dummy_data_n") {
      INT value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_INT);
      cm_msg(MINFO, "SciFiFEB::on_settings_changed", "Set dummy_data_n to %d", value);
      _this->setDummyData_Count(SciFiFEB::FPGA_broadcast_ID,value);
   }
   if (std::string(key.name) == "prbs_decode_bypass") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      cm_msg(MINFO, "SciFiFEB::on_settings_changed", "Set prbs_decode_bypass to %d", value);
      _this->setPRBSDecoder(SciFiFEB::FPGA_broadcast_ID,value);
   }
   if (std::string(key.name) == "mask") {
      BOOL barray[16];
      INT  barraysize=sizeof(barray);
      db_get_data(hDB, hKey, &barray, &barraysize, TID_BOOL);
      for(int i=0;i<16;i++){
           cm_msg(MINFO, "SciFiFEB::on_settings_changed", "Set mask[%d] %d",i, barray[i]);
           _this->setMask(i,barray[i]);
      }
   }
   if (std::string(key.name) == "reset_datapath") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      if(value){
         cm_msg(MINFO, "SciFiFEB::on_settings_changed", "reset_datapath");
         _this->DataPathReset(SciFiFEB::FPGA_broadcast_ID);
         value = FALSE; // reset flag in ODB
         db_set_data(hDB, hKey, &value, sizeof(value), 1, TID_BOOL);
      }
   }
   if (std::string(key.name) == "reset_asics") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      if(value){
         cm_msg(MINFO, "SciFiFEB::on_settings_changed", "reset_asics");
         _this->chipReset(SciFiFEB::FPGA_broadcast_ID);
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
	return (reg_in & (((1<<length)-1)<<offset)) | ((value & ((1<<length)-1))<<offset);
}


//SciFiFEB registers and functions

/**
* Use emulated mutric on fpga for config
*/
void SciFiFEB::setDummyConfig(int FPGA_ID, bool dummy){
	printf("SciFiFEB::setDummyConfig(%d)=%d\n",FPGA_ID,dummy);
	uint32_t val;
	m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
        printf("SciFiFEB(%d)::FE_DUMMYCTRL_REG readback=%8.8x\n",FPGA_ID,val);
        val=reg_setBit(val,0,dummy);
	printf("SciFiFEB(%d)::FE_DUMMYCTRL_REG new=%8.8x\n",FPGA_ID,val);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
}

/**
* use mutrig data emulator on fpga
* n:    number of events per frame
* fast: enable fast mode for data generator (shorter events)
*/

void SciFiFEB::setDummyData_Enable(int FPGA_ID, bool dummy)
{
	printf("SciFiFEB::setDummyData_Enable(%d)=%d\n",FPGA_ID,dummy);
	uint32_t val;
	m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
        printf("SciFiFEB(%d)::FE_DUMMYCTRL_REG readback=%8.8x\n",FPGA_ID,val);
        val=reg_setBit(val,1,dummy);
	printf("SciFiFEB(%d)::FE_DUMMYCTRL_REG new=%8.8x\n",FPGA_ID,val);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
}

void SciFiFEB::setDummyData_Fast(int FPGA_ID, bool fast)
{
	printf("SciFiFEB::setDummyData_Fast(%d)=%d\n",FPGA_ID,fast);
	uint32_t  val;
	m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
        printf("SciFiFEB(%d)::FE_DUMMYCTRL_REG readback=%8.8x\n",FPGA_ID,val);
        val=reg_setBit(val,2,fast);
        printf("SciFiFEB(%d)::FE_DUMMYCTRL_REG new=%8.8x\n",FPGA_ID,val);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
}

void SciFiFEB::setDummyData_Count(int FPGA_ID, int n)
{
        if(n > 255) n = 255;
	printf("SciFiFEB::setDummyData_Count(%d)=%d\n",FPGA_ID,n);
	uint32_t  val;
	m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
        printf("SciFiFEB(%d)::FE_DUMMYCTRL_REG readback=%8.8x\n",FPGA_ID,val);
	val=reg_setRange(val, 9, 3, n);
        printf("SciFiFEB(%d)::FE_DUMMYCTRL_REG new=%8.8x\n",FPGA_ID,val);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
}

/**
* Disable data from specified ASIC
*/
void SciFiFEB::setMask(int asic, bool value){
	printf("SciFiFEB::setMask(%d)=%d (Mapped to %d:%d)\n",asic,value,FPGAid_from_ID(asic),ASICid_from_ID(asic));
	uint32_t val;
	m_mu.FEBsc_read(FPGAid_from_ID(asic), &val, 1 , (uint32_t) FE_DPCTRL_REG);
        printf("SciFiFEB(%d)::FE_DPCTRL_REG readback=%8.8x\n",FPGAid_from_ID(asic),val);
        val=reg_setBit(val,ASICid_from_ID(asic),value);
        printf("SciFiFEB(%d)::FE_DPCTRL_REG new=%8.8x\n",FPGAid_from_ID(asic),val);
	m_mu.FEBsc_write(FPGAid_from_ID(asic), &val, 1 , (uint32_t) FE_DPCTRL_REG);
}

/**
* Disable data from specified ASIC
*/
void SciFiFEB::setPRBSDecoder(uint32_t FPGA_ID, bool enable){
	uint32_t val;
	m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DPCTRL_REG);
        printf("SciFiFEB(%d)::FE_DPCTRL_REG readback=%8.8x\n",FPGA_ID,val);
        val=reg_setBit(val,31,enable);
        printf("SciFiFEB(%d)::FE_DPCTRL_REG new=%8.8x\n",FPGA_ID,val);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DPCTRL_REG);
}


//reset all asics (digital part, CC, fsms, etc.)
void SciFiFEB::chipReset(int FPGA_ID){
	uint32_t val;
	m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG);
	//constant reset should not happen...
	assert(!GET_FE_SUBDET_REST_BIT_CHIP(val));
	//set and clear reset
        val=reg_setBit(val,0,false);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG);
        sleep(1);
        val=reg_setBit(val,0,false);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG);
}

//reset full datapath upstream from merger
void SciFiFEB::DataPathReset(int FPGA_ID){
	uint32_t val;
	m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG);
	//constant reset should not happen...
	assert(!GET_FE_SUBDET_REST_BIT_DPATH(val));
	//set and clear reset
        val=reg_setBit(val,1,true);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG);
        sleep(1);
        val=reg_setBit(val,1,false);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG);
}


