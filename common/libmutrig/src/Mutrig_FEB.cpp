/********************************************************************\

  Name:         Mutrig_FEB.h
  Created by:   Konrad Briggl

Contents:       Definition of functions to talk to a mutrig-based FEB. Designed to no be derived into a SciFi_FEB and a Tiles_FEB class where subdetector specific actions are defined.
		Here: Definition of basic things for mutrig-configuration & datapath settings

\********************************************************************/

#include "Mutrig_FEB.h"
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
#define FE_RESETSKEW_GLOBALS_REG  (SC_REG_OFFSET+0xb)
#define FE_SPIDATA_ADDR		0

//status flags from FEB
#define FEB_REPLY_SUCCESS 0
#define FEB_REPLY_ERROR   1

const uint8_t MutrigFEB::FPGA_broadcast_ID=0;

//ASIC configuration:
//Configure all asics under prefix (e.g. prefix="/Equipment/SciFi")
int MutrigFEB::ConfigureASICs(HNDLE hDB, const char* equipment_name, const char* odb_prefix){
   printf("MutrigFEB::ConfigureASICs()\n");
   int status = mutrig::midasODB::MapForEach(hDB,odb_prefix,[this,&odb_prefix,&equipment_name](mutrig::MutrigConfig* config, int asic){
      int status=SUCCESS;
      uint32_t reg;
      cm_msg(MINFO, "setup_mutrig" , "Configuring MuTRiG asic %s/Settings/ASICs/%i/", odb_prefix, asic);
      uint32_t rpc_status;
      try {
         //Write ASIC number & Configuraton
	 rpc_status=m_mu.FEBsc_NiosRPC(FPGAid_from_ID(asic),0x0110,{{reinterpret_cast<uint32_t*>(&asic),1},{reinterpret_cast<uint32_t*>(config->bitpattern_w), config->length_32bits}});
      } catch(std::exception& e) {
          cm_msg(MERROR, "setup_mutrig", "Communication error while configuring MuTRiG %d: %s", asic, e.what());
          set_equipment_status(equipment_name, "SB-FEB Communication error", "red");
          return FE_ERR_HW; //note: return of lambda function
      }
      if(rpc_status!=FEB_REPLY_SUCCESS){
         //configuration mismatch, report and break foreach-loop
         set_equipment_status(equipment_name,  "MuTRiG config failed", "red");
         cm_msg(MERROR, "setup_mutrig", "MuTRiG configuration error for ASIC %i", asic);
         return FE_ERR_HW;//note: return of lambda function
      }
      return FE_SUCCESS;//note: return of lambda function
   });//MapForEach
   return status; //status of foreach function, SUCCESS when no error.
   return 0;
}

//MIDAS callback function for FEB register Setter functions
void MutrigFEB::on_settings_changed(HNDLE hDB, HNDLE hKey, INT, void * userdata)
{
   MutrigFEB* _this=static_cast<MutrigFEB*>(userdata);
   KEY key;
   db_get_key(hDB, hKey, &key);
   printf("MutrigFEB::on_settings_changed(%s)\n",key.name);
   if (std::string(key.name) == "dummy_config") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      cm_msg(MINFO, "MutrigFEB::on_settings_changed", "Set dummy_config to %d", value);
      _this->setDummyConfig(MutrigFEB::FPGA_broadcast_ID,value);
   }
   if (std::string(key.name) == "dummy_data") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      cm_msg(MINFO, "MutrigFEB::on_settings_changed", "Set dummy_data to %d", value);
      _this->setDummyData_Enable(MutrigFEB::FPGA_broadcast_ID,value);
   }
   if (std::string(key.name) == "dummy_data_fast") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      cm_msg(MINFO, "MutrigFEB::on_settings_changed", "Set dummy_data_fast to %d", value);
      _this->setDummyData_Fast(MutrigFEB::FPGA_broadcast_ID,value);
   }
   if (std::string(key.name) == "dummy_data_n") {
      INT value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_INT);
      cm_msg(MINFO, "MutrigFEB::on_settings_changed", "Set dummy_data_n to %d", value);
      _this->setDummyData_Count(MutrigFEB::FPGA_broadcast_ID,value);
   }
   if (std::string(key.name) == "prbs_decode_disable") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      cm_msg(MINFO, "MutrigFEB::on_settings_changed", "Set prbs_decode_disable to %d", value);
      _this->setPRBSDecoderDisable(MutrigFEB::FPGA_broadcast_ID,value);
   }
   if (std::string(key.name) == "LVDS_waitforall") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      cm_msg(MINFO, "MutrigFEB::on_settings_changed", "Set LVDS_waitforall to %d", value);
      _this->setPRBSDecoderDisable(MutrigFEB::FPGA_broadcast_ID,value);
   }
   if (std::string(key.name) == "LVDS_waitforall_sticky") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      cm_msg(MINFO, "MutrigFEB::on_settings_changed", "Set LVDS_waitforall_sticky to %d", value);
      _this->setPRBSDecoderDisable(MutrigFEB::FPGA_broadcast_ID,value);
   }
   if (std::string(key.name) == "mask") {
      BOOL barray[16];
      INT  barraysize=sizeof(barray);
      db_get_data(hDB, hKey, &barray, &barraysize, TID_BOOL);
      for(int i=0;i<16;i++){
           cm_msg(MINFO, "MutrigFEB::on_settings_changed", "Set mask[%d] %d",i, barray[i]);
           _this->setMask(i,barray[i]);
      }
   }
   if (std::string(key.name) == "reset_datapath") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      if(value){
         cm_msg(MINFO, "MutrigFEB::on_settings_changed", "reset_datapath");
         _this->DataPathReset(MutrigFEB::FPGA_broadcast_ID);
         value = FALSE; // reset flag in ODB
         db_set_data(hDB, hKey, &value, sizeof(value), 1, TID_BOOL);
      }
   }
   if (std::string(key.name) == "reset_asics") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      if(value){
         cm_msg(MINFO, "MutrigFEB::on_settings_changed", "reset_asics");
         _this->chipReset(MutrigFEB::FPGA_broadcast_ID);
         value = FALSE; // reset flag in ODB
         db_set_data(hDB, hKey, &value, sizeof(value), 1, TID_BOOL);
      }
   }
   
   //reset skew settings
   if (std::string(key.name) == "resetskew_cphase") {
        cm_msg(MINFO, "MutrigFEB::on_settings_changed", "Updating reset skew cphase settings");
        BOOL barray[4];
        INT  barraysize=sizeof(barray);
        db_get_data(hDB, hKey, &barray, &barraysize, TID_BOOL);
	_this->setResetSkewCphase(MutrigFEB::FPGA_broadcast_ID,barray);
   }
   if (std::string(key.name) == "resetskew_cdelay") {
        cm_msg(MINFO, "MutrigFEB::on_settings_changed", "Updating reset skew cdelay settings");
        BOOL barray[4];
        INT  barraysize=sizeof(barray);
        db_get_data(hDB, hKey, &barray, &barraysize, TID_BOOL);
	_this->setResetSkewCdelay(MutrigFEB::FPGA_broadcast_ID,barray);
   }
   if (std::string(key.name) == "resetskew_phases") {
        cm_msg(MINFO, "MutrigFEB::on_settings_changed", "Updating reset skew phase settings");
        INT array[4];
        INT arraysize=sizeof(array);
        db_get_data(hDB, hKey, &array, &arraysize, TID_INT);
	_this->setResetSkewPhases(MutrigFEB::FPGA_broadcast_ID,array);
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


//MutrigFEB registers and functions

/**
* Use emulated mutric on fpga for config
*/
void MutrigFEB::setDummyConfig(int FPGA_ID, bool dummy){
	printf("MutrigFEB::setDummyConfig(%d)=%d\n",FPGA_ID,dummy);
	uint32_t val;

        //TODO: shadowing should know about broadcast FPGA ID
	//TODO: implement pull from FPGA when shadow value is not stored
	//m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
        //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG readback=%8.8x\n",FPGA_ID,val);
        val=m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG];

        val=reg_setBit(val,0,dummy);
	printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG new=%8.8x\n",FPGA_ID,val);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG, m_ask_sc_reply);
	m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG]=val;
}

/**
* use mutrig data emulator on fpga
* n:    number of events per frame
* fast: enable fast mode for data generator (shorter events)
*/

void MutrigFEB::setDummyData_Enable(int FPGA_ID, bool dummy)
{
	printf("MutrigFEB::setDummyData_Enable(%d)=%d\n",FPGA_ID,dummy);
	uint32_t val;

        val=m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG];
	//m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
        //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG readback=%8.8x\n",FPGA_ID,val);

        val=reg_setBit(val,1,dummy);
	printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG new=%8.8x\n",FPGA_ID,val);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG, m_ask_sc_reply);
	m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG]=val;
}

void MutrigFEB::setDummyData_Fast(int FPGA_ID, bool fast)
{
	printf("MutrigFEB::setDummyData_Fast(%d)=%d\n",FPGA_ID,fast);
	uint32_t  val;
        val=m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG];
	//m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
        //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG readback=%8.8x\n",FPGA_ID,val);

        val=reg_setBit(val,2,fast);
        printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG new=%8.8x\n",FPGA_ID,val);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG, m_ask_sc_reply);
	m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG]=val;
}

void MutrigFEB::setDummyData_Count(int FPGA_ID, int n)
{
        if(n > 255) n = 255;
	printf("MutrigFEB::setDummyData_Count(%d)=%d\n",FPGA_ID,n);
	uint32_t  val;
        val=m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG];
	//m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
        //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG readback=%8.8x\n",FPGA_ID,val);
	val=reg_setRange(val, 9, 3, n);

        printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG new=%8.8x\n",FPGA_ID,val);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG,m_ask_sc_reply);
	m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG]=val;
}

/**
* Disable data from specified ASIC
*/
void MutrigFEB::setMask(int asic, bool value){
	printf("MutrigFEB::setMask(%d)=%d (Mapped to %d:%d)\n",asic,value,FPGAid_from_ID(asic),ASICid_from_ID(asic));
	uint32_t val;
        val=m_reg_shadow[FPGAid_from_ID(asic)][FE_DPCTRL_REG];
	//m_mu.FEBsc_read(FPGAid_from_ID(asic), &val, 1 , (uint32_t) FE_DPCTRL_REG);
        //printf("MutrigFEB(%d)::FE_DPCTRL_REG readback=%8.8x\n",FPGAid_from_ID(asic),val);

        val=reg_setBit(val,ASICid_from_ID(asic),value);
        //printf("MutrigFEB(%d)::FE_DPCTRL_REG new=%8.8x\n",FPGAid_from_ID(asic),val);
	m_mu.FEBsc_write(FPGAid_from_ID(asic), &val, 1 , (uint32_t) FE_DPCTRL_REG,m_ask_sc_reply);
        m_reg_shadow[FPGAid_from_ID(asic)][FE_DPCTRL_REG]=val;
}



/**
* Disable prbs decoder in FPGA
*/
void MutrigFEB::setPRBSDecoderDisable(uint32_t FPGA_ID, bool disable){
	printf("MutrigFEB::setPRBSDecoderDisable(%d)=%d\n",FPGA_ID,disable);
	uint32_t val;
        val=m_reg_shadow[FPGA_ID][FE_DPCTRL_REG];
	//m_mu.FEBsc_read(FPGAid_from_ID(asic), &val, 1 , (uint32_t) FE_DPCTRL_REG);
        //printf("MutrigFEB(%d)::FE_DPCTRL_REG readback=%8.8x\n",FPGAid_from_ID(asic),val);

        val=reg_setBit(val,31,disable);
        //printf("MutrigFEB(%d)::FE_DPCTRL_REG new=%8.8x\n",FPGA_ID,val);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DPCTRL_REG,m_ask_sc_reply);
        m_reg_shadow[FPGA_ID][FE_DPCTRL_REG]=val;
}

void MutrigFEB::setWaitForAll(uint32_t FPGA_ID, bool value){
	printf("MutrigFEB::setWaitForAll(%d)=%d\n",FPGA_ID,value);
	uint32_t val;
        val=m_reg_shadow[FPGA_ID][FE_DPCTRL_REG];
	//m_mu.FEBsc_read(FPGAid_from_ID(asic), &val, 1 , (uint32_t) FE_DPCTRL_REG);
        //printf("MutrigFEB(%d)::FE_DPCTRL_REG readback=%8.8x\n",FPGAid_from_ID(asic),val);

        val=reg_setBit(val,30,value);
        //printf("MutrigFEB(%d)::FE_DPCTRL_REG new=%8.8x\n",FPGAid_from_ID(asic),val);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DPCTRL_REG,m_ask_sc_reply);
        m_reg_shadow[FPGA_ID][FE_DPCTRL_REG]=val;
}

void MutrigFEB::setWaitForAllSticky(uint32_t FPGA_ID, bool value){
	printf("MutrigFEB::setWaitForAllSticky(%d)=%d\n",FPGA_ID,value);
	uint32_t val;
        val=m_reg_shadow[FPGA_ID][FE_DPCTRL_REG];
	//m_mu.FEBsc_read(FPGAid_from_ID(asic), &val, 1 , (uint32_t) FE_DPCTRL_REG);
        //printf("MutrigFEB(%d)::FE_DPCTRL_REG readback=%8.8x\n",FPGAid_from_ID(asic),val);

        val=reg_setBit(val,29,value);
        //printf("MutrigFEB(%d)::FE_DPCTRL_REG new=%8.8x\n",FPGAid_from_ID(asic),val);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DPCTRL_REG,m_ask_sc_reply);
        m_reg_shadow[FPGA_ID][FE_DPCTRL_REG]=val;
}



//reset all asics (digital part, CC, fsms, etc.)
void MutrigFEB::chipReset(int FPGA_ID){
	uint32_t val=0;
	//m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG);
	//constant reset should not happen...
	//assert(!GET_FE_SUBDET_REST_BIT_CHIP(val));
	//set and clear reset
        val=reg_setBit(val,0,true);
        printf("writing %x\n",val);
	m_mu.FEBsc_write(FPGA_ID, &val, 0 , (uint32_t) FE_SUBDET_RESET_REG,m_ask_sc_reply);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        val=reg_setBit(val,0,false);
        printf("writing %x\n",val);
	m_mu.FEBsc_write(FPGA_ID, &val, 0 , (uint32_t) FE_SUBDET_RESET_REG,m_ask_sc_reply);
}

//reset full datapath upstream from merger
void MutrigFEB::DataPathReset(int FPGA_ID){
	uint32_t val=0;
	//m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG);
	//constant reset should not happen...
	//assert(!GET_FE_SUBDET_REST_BIT_DPATH(val));
	//set and clear reset
        val=reg_setBit(val,1,true);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG,m_ask_sc_reply);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        val=reg_setBit(val,1,false);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG,m_ask_sc_reply);
}

//set reset skew configuration
void MutrigFEB::setResetSkewCphase(int FPGA_ID, BOOL cphase[4]){
        uint32_t val=m_reg_shadow[FPGA_ID][FE_RESETSKEW_GLOBALS_REG];
        for(int i=0;i<4;i++){
            val=reg_setBit(val,i+6,cphase[i]);
        }
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_RESETSKEW_GLOBALS_REG, m_ask_sc_reply);
        m_reg_shadow[FPGA_ID][FE_RESETSKEW_GLOBALS_REG]=val;
}

void MutrigFEB::setResetSkewCdelay(int FPGA_ID, BOOL cdelay[4]){
        uint32_t val=m_reg_shadow[FPGA_ID][FE_RESETSKEW_GLOBALS_REG];
        for(int i=0;i<4;i++){
            val=reg_setBit(val,i+10,cdelay[i]);
        }
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_RESETSKEW_GLOBALS_REG, m_ask_sc_reply);
        m_reg_shadow[FPGA_ID][FE_RESETSKEW_GLOBALS_REG]=val;
}

void MutrigFEB::setResetSkewPhases(int FPGA_ID, INT phases[4]){
	uint32_t val[4];
        for(int i=0;i<4;i++){
        	val[i]=phases[i];
        }
	m_mu.FEBsc_NiosRPC(FPGA_ID, 0x0104, {{val,4}});
}
