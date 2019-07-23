/********************************************************************\

  Name:         FEB_access.h
  Created by:   Konrad Briggl

Contents:       Definition of fumctions in namespace mudaq::mutrig
		to provide an abstraction layer to the (slow control) functions on the FE-FPGA

\********************************************************************/

#include "SciFi_FEB.h"
#include "SciFi_FEB_registers.h"
#include "midas.h"
#include "mfe.h" //for set_equipment_status

#include "mudaq_device.h"
#include "mutrig_config.h"
#include "mutrig_midasodb.h"
#include <thread>
#include <chrono>
namespace mudaq { namespace mutrig {

//ASIC configuration:
//Configure all asics under prefix (e.g. prefix="/Equipment/SciFi")
int FEB::ConfigureASICs(HNDLE hDB, const char* equipment_name, const char* odb_prefix){
   int status = mudaq::mutrig::midasODB::MapForEach(hDB,odb_prefix,[this,&odb_prefix,&equipment_name](mudaq::mutrig::Config* config, int asic){
   int status=SUCCESS;
   //try each asic twice, i.e. give up when cnt>1
   int cnt = 0;
   while(cnt<1) {
      try {
         cm_msg(MINFO, "setup_mutrig" , "Configuring MuTRiG asic %s/ASICs/%i/", odb_prefix, asic);
	 uint32_t FPGA_ID=asic/nAsicsPerFrontend; //TODO: check mapping makes sense
	 //Write configuration
	 m_mu.FEBsc_write(FPGA_ID, reinterpret_cast<uint32_t*>(config->bitpattern_w), config->length_32bits , (uint32_t) FE_SPIDATA_ADDR);
	 //Write handleID and start bit to trigger SPI transaction
	 uint32_t data=0;
	 data=SET_FE_SPICTRL_BIT_START(data);
	 data=SET_FE_SPICTRL_CHIPID_RANGE(data,(unsigned int)asic);
	 m_mu.FEBsc_write(FPGA_ID, &data, 1, (uint32_t) FE_SPICTRL_REGISTER);
         //Wait for transaction to finish
         uint cnt = 0;
	 do{
            if(++cnt >= 10000) throw std::runtime_error("SPI transaction timeout while configuring asic"+std::to_string(asic));
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
	    m_mu.FEBsc_read(FPGA_ID, &data, 1, (uint32_t) FE_SPICTRL_REGISTER);

         }while(GET_FE_SPICTRL_BIT_START(data));
	 //Read back configuration
	 m_mu.FEBsc_read(FPGA_ID, reinterpret_cast<uint32_t*>(config->bitpattern_r), config->length_32bits , (uint32_t) FE_SPIDATA_ADDR);

	 status=config->VerifyReadbackPattern();
         status=SUCCESS; //TODO:Remove
         if(status==SUCCESS) break; //configuration good, stopping here. Otherwise try another time without complaining here.
      } catch(std::exception& e) {
         cm_msg(MERROR, "setup_mutrig", "Communication error while configuring MuTRiG %d, try %d: %s", asic,cnt, e.what());
         set_equipment_status(equipment_name, "Communication error while configuring MuTRiG", "red");
         return FE_ERR_HW; //note: return of lambda function
      }
   }
   if(status!=SUCCESS){
      //configuration mismatch, report and break foreach-loop
      set_equipment_status(equipment_name,  "MuTRiG config failed", "red");
      cm_msg(MERROR, "setup_mutrig", "MuTRiG configuration error for ASIC %i at try %d", asic, cnt);
      cm_msg(MERROR, "setup_mutrig", "%s",config->GetVerificationError().c_str());
   }
   return status;//note: return of lambda function
   });//MapForEach
   return status; //status of foreach function, SUCCESS when no error.
}


//FEB registers and functions

/**
* Use emulated mutric on fpga for config
*/
void FEB::setDummyConfig(int FPGA_ID, bool dummy){
	uint32_t val;
	m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
	if(dummy) 
		val=SET_FE_DUMMYCTRL_BIT_SPI(val);
	else      
		val=UNSET_FE_DUMMYCTRL_BIT_SPI(val);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
}

/**
* use mutrig data emulator on fpga
* n:    number of events per frame
* fast: enable fast mode for data generator (shorter events)
*/
void FEB::setDummyData(int FPGA_ID, bool dummy, int n, bool fast){
	uint32_t  val;
	m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
	if(dummy) 
		val=SET_FE_DUMMYCTRL_BIT_DATAGEN(val); 
	else 
		val=UNSET_FE_DUMMYCTRL_BIT_DATAGEN(val);
	if(fast)  
		val=SET_FE_DUMMYCTRL_BIT_SHORTHIT(val); 
	else 
		val=UNSET_FE_DUMMYCTRL_BIT_SHORTHIT(val);

	val=SET_FE_DUMMYCTRL_HITCNT_RANGE(val,(unsigned int) n);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
}

/**
* Disable data from specified ASIC
*/
void FEB::setMask(int ASIC, bool value){
	uint32_t val;
	uint32_t FPGA_ID=ASIC/nAsicsPerFrontend; //TODO:check mapping
	m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DPCTRL_REG);
	if(value) 
			val |=  (1<<ASIC);
	else      
			val &= ~(1<<ASIC);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DPCTRL_REG);
}

/**
* Disable data from specified ASIC
*/
void FEB::setPRBSDecoder(uint32_t FPGA_ID, bool enable){
	uint32_t val;
	m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DPCTRL_REG);
	if(enable) 
			val=SET_FE_DPCTRL_BIT_PRBSDEC(val); 
	else 
			val=UNSET_FE_DPCTRL_BIT_PRBSDEC(val);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DPCTRL_REG);
}


//reset all asics (digital part, CC, fsms, etc.)
void FEB::chipReset(int FPGA_ID){
	uint32_t val;
	m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG);
	//constant reset should not happen...
	assert(!GET_FE_SUBDET_REST_BIT_CHIP(val));
	//set and clear reset
	val=SET_FE_SUBDET_REST_BIT_CHIP(val);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG);
	val=UNSET_FE_SUBDET_REST_BIT_CHIP(val);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG);
}

//reset full datapath upstream from merger
void FEB::DataPathReset(int FPGA_ID){
	uint32_t val;
	m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG);
	//constant reset should not happen...
	assert(!GET_FE_SUBDET_REST_BIT_DPATH(val));
	//set and clear reset
	val=SET_FE_SUBDET_REST_BIT_DPATH(val);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG);
	val=UNSET_FE_SUBDET_REST_BIT_DPATH(val);
	m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG);
}


}//namespace mutrig 
}//namespace mudaq 

