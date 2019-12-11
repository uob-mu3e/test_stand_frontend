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
      std::map<uint16_t,std::map<uint32_t,uint32_t> > m_reg_shadow; /*[FPGA_ID][reg]*/
      bool m_ask_sc_reply;
   public:
      MutrigFEB(const MutrigFEB&)=delete;
      MutrigFEB(mudaq::MudaqDevice& mu):m_mu(mu),m_ask_sc_reply(true){};

      void SetAskSCReply(bool ask){m_ask_sc_reply=ask;};

      //MIDAS callback for all setters below. Made static and using the user data argument as "this" to ease binding to C-style midas-callbacks
      static void on_settings_changed(HNDLE hDB, HNDLE hKey, INT, void *);

      //Write all registers based on ODB values
      int WriteAll(HNDLE hDB, const char* odb_prefix);

      //ASIC configuration:
      //Configure all asics under prefix (e.g. prefix="/Equipment/SciFi"), report any errors as equipment_name
      int ConfigureASICs(HNDLE hDB, const char* equipment_name, const char* odb_prefix);

   protected:
      //Mapping from ASIC number to FPGA_ID and ASIC_ID
      static const uint16_t FPGA_broadcast_ID;
      virtual uint16_t FPGAid_from_ID(int asic)=0;
      virtual uint16_t ASICid_from_ID(int asic)=0;


      //Read counter values from FEB, store in subtree $odb_prefix/Variables/Counters/ 
      int ReadBackCounters(HNDLE hDB, uint16_t FPGA_ID, const char* odb_prefix);


      //FEB registers and functions

      /**
       * Use emulated mutric on fpga for config (NOT IMPLEMENTED IN FW)
       */
      void setDummyConfig(uint16_t FPGA_ID,bool dummy = true);
  
      /**
       * use mutrig data emulator on fpga
       * n:    number of events per frame
       * fast: enable fast mode for data generator (shorter events)
       */
      void setDummyData_Enable(uint16_t FPGA_ID, bool dummy = true);
      void setDummyData_Count(uint16_t FPGA_ID, int n = 255);
      void setDummyData_Fast(uint16_t FPGA_ID, bool fast = false);
  
      /**
       * Disable data from specified ASIC
       */
      void setMask(int ASIC, bool value); //ASIC: global ASIC ID
  
      /**
       * Disable PRBS decoder in FPGA
       */
      void setPRBSDecoderDisable(uint32_t FPGA_ID,bool disable);

      /**
       * Wait for lvds receivers ready strategy
       */
      void setWaitForAll(uint32_t FPGA_ID,bool val);
      void setWaitForAllSticky(uint32_t FPGA_ID,bool val);



      void syncReset(uint16_t FPGA_ID){chipReset(FPGA_ID);}; //should be resetting the ASICs coarse counter only, missing pin on the asic. For future use
      void chipReset(uint16_t FPGA_ID); //reset all asics (digital part, CC, fsms, etc.)
      void DataPathReset(uint16_t FPGA_ID); //in FE-FPGA: everything upstream of merger (in the stream path)
      void LVDS_RX_Reset(uint16_t FPGA_ID); //in FE-FPGA: LVDS receiver blocks


      //reset signal alignment control
      void setResetSkewCphase(uint16_t FPGA_ID, BOOL cphase[4]);
      void setResetSkewCdelay(uint16_t FPGA_ID, BOOL cdelay[4]);
      void setResetSkewPhases(uint16_t FPGA_ID, INT phases[4]);

};//class MutrigFEB

#endif // MUTRIG_FEB_H
