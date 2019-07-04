/********************************************************************\

  Name:         FEB_access.h
  Created by:   Konrad Briggl

Contents:       Definition of fumctions in namespace mudaq::mutrig
		to provide an abstraction layer to the (slow control) functions on the FE-FPGA

\********************************************************************/

#ifndef FEB_ACCESS_H
#define FEB_ACCESS_H

#include "midas.h"
#include "mudaq_device.h"
#include "mutrig_config.h"

namespace mudaq { namespace mutrig {
class FEB {
   private:
      mudaq::MudaqDevice& m_mu;
      uint32_t m_pcie_mem_start;
   public:
      FEB(const FEB&)=delete;	   
      FEB(mudaq::MudaqDevice& mu, uint32_t pcie_mem_start):m_mu(mu),m_pcie_mem_start(pcie_mem_start){};

      //ASIC configuration:
      //Configure all asics under prefix (e.g. prefix="/Equipment/SciFi")
      int ConfigureASICs(HNDLE hDB, const char* equipment_name, const char* odb_prefix);

      //FEB registers and functions

      /**
       * Use emulated mutric on fpga for config
       */
      void setDummyConfig(int FPGA_ID,bool dummy = true);
  
      /**
       * use mutrig data emulator on fpga
       * n:    number of events per frame
       * fast: enable fast mode for data generator (shorter events)
       */
      void setDummyData(int FPGA_ID, bool dummy = true, int n = 255, bool fast = false);
  
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
}//namespace mutrig 
}//namespace mudaq 

#endif // FEB_ACCESS_H
