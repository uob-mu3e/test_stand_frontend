/********************************************************************\

  Name:         FEB_access.h
  Created by:   Konrad Briggl

Contents:       Definition of fumctions in namespace mudaq::mutrig
		to provide an abstraction layer to the (slow control) functions on the FE-FPGA

\********************************************************************/

#ifndef FEB_ACCESS_H
#define FEB_ACCESS_H

#include "midas.h"
#include "mudaq_device_scifi.h"
#include "mutrig_config.h"

class SciFiFEB {
   private:
      mudaq::MudaqDevice& m_mu;
      static SciFiFEB* m_instance; //signleton instance pointer
      SciFiFEB(const SciFiFEB&)=delete;
      SciFiFEB(mudaq::MudaqDevice& mu):m_mu(mu){};
      std::map<uint8_t,std::map<uint32_t,uint32_t> > m_reg_shadow; /*[FPGA_ID][reg]*/
   public:
      static const uint8_t FPGA_broadcast_ID;

      static SciFiFEB* Create(mudaq::MudaqDevice& mu){printf("FEB::Create()");if(!m_instance) m_instance=new SciFiFEB(mu); return m_instance;};
      static SciFiFEB* Instance(){return m_instance;};

      //Mapping from ASIC number to FPGA_ID and ASIC_ID
      uint8_t FPGAid_from_ID(int asic);
      uint8_t ASICid_from_ID(int asic);
      //ASIC configuration:
      //Configure all asics under prefix (e.g. prefix="/Equipment/SciFi")
      int ConfigureASICs(HNDLE hDB, const char* equipment_name, const char* odb_prefix);

      //FEB registers and functions

      //MIDAS callback for all setters below. made static and using the user data argument as "this" to ease binding to C-style midas-callbacks
      static void on_settings_changed(HNDLE hDB, HNDLE hKey, INT, void *);

      /**
       * Use emulated mutric on fpga for config
       */
      void setDummyConfig(int FPGA_ID,bool dummy = true);
  
      /**
       * use mutrig data emulator on fpga
       * n:    number of events per frame
       * fast: enable fast mode for data generator (shorter events)
       */
      void setDummyData_Enable(int FPGA_ID, bool dummy = true);
      void setDummyData_Count(int FPGA_ID, int n = 255);
      void setDummyData_Fast(int FPGA_ID, bool fast = false);
  
      /**
       * Disable data from specified ASIC
       */
      void setMask(int ASIC, bool value);
  
      /**
       * Disable data from specified ASIC
       */
      void setPRBSDecoder(uint32_t FPGA_ID,bool enable);


      void syncReset(int FPGA_ID){chipReset(FPGA_ID);}; //should be resetting the ASICs coarse counter only, missing pin on the asic. For future use
      void chipReset(int FPGA_ID); //reset all asics (digital part, CC, fsms, etc.)
      void DataPathReset(int FPGA_ID); //in FE-FPGA: everything upstream of merger (in the stream path)
      //TODO: add more resets for FE-FPGA blocks

};//class FEB

#endif // FEB_ACCESS_H
