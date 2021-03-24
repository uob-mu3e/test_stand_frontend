/********************************************************************\

  Name:         Mutrig_FEB.h
  Created by:   Konrad Briggl

Contents:       Definition of functions to talk to a mutrig-based FEB. Designed to no be derived into a SciFi_FEB and a Tiles_FEB class where subdetector specific actions are defined.
		Here: Definition of basic things for mutrig-configuration & datapath settings

\********************************************************************/

#ifndef MUTRIG_FEB_H
#define MUTRIG_FEB_H

#include "midas.h"
#include "odbxx.h"
#include "FEBSlowcontrolInterface.h"
#include "MutrigConfig.h"
#include "MuFEB.h"

using midas::odb;

class MutrigFEB : public MuFEB{
   protected:
      std::map<uint16_t,std::map<uint32_t,uint32_t> > m_reg_shadow; /*[FPGA_ID][reg]*/

   public:
      MutrigFEB(const MutrigFEB&)=delete;
      MutrigFEB(FEBSlowcontrolInterface & feb_sc_,
            const vector<mappedFEB> & febs_,
            const uint64_t & febmask_,
            const char* equipment_name_,
            const char* odb_prefix_,
            const uint8_t SB_number_):
      MuFEB(feb_sc_, febs_, febmask_, equipment_name_, odb_prefix_, SB_number_)
      {}
      virtual ~MutrigFEB(){}
      uint16_t GetNumASICs() const {return febs.size()*GetModulesPerFEB()*GetASICSPerModule();}
      uint16_t GetNumModules() const {return febs.size()*GetModulesPerFEB();}

      //MIDAS callback for all setters below (DAQ related, mapped to functions on FEB / settings from the DAQ subdirectory).
      //Made static and using the user data argument as "this" to ease binding to C-style midas-callbacks
      static void on_settings_changed(odb o, void * userdata);

      //Write all registers based on ODB values
      int WriteAll();

      //ASIC configuration:
      //Configure all asics under prefix (e.g. prefix="/Equipment/SciFi"), report any errors as equipment_name
      int ConfigureASICs();

      //Read counter values from FEB, store in subtree $odb_prefix/Variables/Counters/
      //Parameter FPGA_ID refers to global numbering, i.e. before mapping
      int ReadBackCounters(uint16_t FPGA_ID);
      void ReadBackAllCounters(){for(size_t i=0;i<febs.size();i++) ReadBackCounters(i);}
      int ResetCounters(uint16_t FPGA_ID);
      void ResetAllCounters(){for(size_t i=0;i<febs.size();i++) ResetCounters(i);}


      //Read datapath status values from FEB, store in subtree $odb_prefix/Variables/FEB datapath status
      //Parameter FPGA_ID refers to global numbering, i.e. before mapping
      int ReadBackDatapathStatus(uint16_t FPGA_ID);
      void ReadBackAllDatapathStatus(){for(size_t i=0;i<febs.size();i++) ReadBackDatapathStatus(i);}


   protected:

      //Foreach loop over all asics under this prefix. Call with a lambda function,
      //e.g. midasODB::MapForEach(hDB, "/Equipment/SciFi",[mudaqdev_ptr](Config c,int asic){mudaqdev_ptr->ConfigureAsic(c,asic);});
      //Function must return SUCCESS, otherwise loop is stopped.
      int MapForEach(std::function<int(mutrig::MutrigConfig* /*mutrig config*/,int /*ASIC #*/)> func);




      //FEB registers and functions.
      //Parameter FPGA_ID refers to the physical FEB port, i.e. after mapping
      //Parameter ASIC refers to the global numbering scheme, i.e. before mapping

      /**
       * Use emulated mutric on fpga for config
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
       * Disable data from specified ASIC (asic number in global numbering scheme, i.e. before mapping)
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



      void syncReset(uint16_t FPGA_ID){chipReset(FPGA_ID);} //should be resetting the ASICs coarse counter only, missing pin on the asic. For future use
      void chipReset(uint16_t FPGA_ID); //reset all asics (digital part, CC, fsms, etc.)
      void DataPathReset(uint16_t FPGA_ID); //in FE-FPGA: everything upstream of merger (in the stream path)
      void LVDS_RX_Reset(uint16_t FPGA_ID); //in FE-FPGA: LVDS receiver blocks


      //reset signal alignment control
      void setResetSkewCphase(uint16_t FPGA_ID, BOOL cphase[]);
      void setResetSkewCdelay(uint16_t FPGA_ID, BOOL cdelay[]);
      void setResetSkewPhases(uint16_t FPGA_ID, INT phases[]);

};//class MutrigFEB

#endif // MUTRIG_FEB_H
