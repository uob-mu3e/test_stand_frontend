/********************************************************************\

  Name:         Mutrig_FEB.h
  Created by:   Konrad Briggl

Contents:       Definition of functions to talk to a mutrig-based FEB. Designed to no be derived into a SciFi_FEB and a Tiles_FEB class where subdetector specific actions are defined.
		Here: Definition of basic things for mutrig-configuration & datapath settings

\********************************************************************/

#ifndef MUTRIG_FEB_H
#define MUTRIG_FEB_H

#include "midas.h"
#include "mudaq_device_scifi.h"
#include "mutrig_config.h"

class MutrigFEB {
   protected:
      mudaq::MudaqDevice& m_mu;
      std::map<uint8_t,std::map<uint32_t,uint32_t> > m_reg_shadow; /*[FPGA_ID][reg]*/
      bool m_ask_sc_reply;
   public:
      MutrigFEB(const MutrigFEB&)=delete;
      MutrigFEB(mudaq::MudaqDevice& mu):m_mu(mu),m_ask_sc_reply(true){};

      void SetAskSCReply(bool ask){m_ask_sc_reply=ask;};

      //Mapping from ASIC number to FPGA_ID and ASIC_ID
      static const uint8_t FPGA_broadcast_ID;
      virtual uint8_t FPGAid_from_ID(int asic)=0;
      virtual uint8_t ASICid_from_ID(int asic)=0;

      //ASIC configuration:
      //Configure all asics under prefix (e.g. prefix="/Equipment/SciFi")
      int ConfigureASICs(HNDLE hDB, const char* equipment_name, const char* odb_prefix);

      //FEB registers and functions

      //MIDAS callback for all setters below. Made static and using the user data argument as "this" to ease binding to C-style midas-callbacks
      static void on_settings_changed(HNDLE hDB, HNDLE hKey, INT, void *);

      /**
       * Use emulated mutric on fpga for config (NOT IMPLEMENTED IN FW)
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
       * Disable PRBS decoder in FPGA
       */
      void setPRBSDecoderDisable(uint32_t FPGA_ID,bool disable);

      /**
       * Wait for lvds receivers ready strategy
       */
      void setWaitForAll(uint32_t FPGA_ID,bool val);
      void setWaitForAllSticky(uint32_t FPGA_ID,bool val);



      void syncReset(int FPGA_ID){chipReset(FPGA_ID);}; //should be resetting the ASICs coarse counter only, missing pin on the asic. For future use
      void chipReset(int FPGA_ID); //reset all asics (digital part, CC, fsms, etc.)
      void DataPathReset(int FPGA_ID); //in FE-FPGA: everything upstream of merger (in the stream path)
      //TODO: add more resets for FE-FPGA blocks

};//class MutrigFEB

#endif // MUTRIG_FEB_H
