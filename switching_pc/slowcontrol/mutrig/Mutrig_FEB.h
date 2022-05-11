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
      std::string odb_prefix;
   public:
      MutrigFEB(const MutrigFEB&)=delete;
      MutrigFEB(FEBSlowcontrolInterface & feb_sc_,
            const vector<mappedFEB> & febs_,
            const uint64_t & febmask_,
            std::string equipment_name_,
            std::string link_equipment_name_,
            std::string odb_prefix_,
            const uint8_t SB_number_):
      MuFEB(feb_sc_, febs_, febmask_, equipment_name_, link_equipment_name_, SB_number_),odb_prefix(odb_prefix_)
      {}
      virtual ~MutrigFEB(){}
      uint16_t GetNumASICs() const {return febs.size()*GetModulesPerFEB()*GetASICSPerModule();}
      uint16_t GetNumModules() const {return febs.size()*GetModulesPerFEB();}

      //MIDAS callback for all setters below (DAQ related, mapped to functions on FEB / settings from the DAQ subdirectory).
      //Made static and using the user data argument as "this" to ease binding to C-style midas-callbacks
      static void on_settings_changed(odb o, void * userdata);
      static void on_commands_changed(odb o, void * userdata);

      //Write all registers based on ODB values
      int WriteAll(uint32_t nasics);

      //ASIC configuration:
      //Configure all asics under prefix (e.g. prefix="/Equipment/SciFi"), report any errors as equipment_name
      int ConfigureASICs();
      int ChangeTDCTest(bool o);
      int ConfigureASICsAllOff();

      //Read counter values from FEB, store in subtree $odb_prefix/Variables/Counters/
      //Parameter FPGA_ID refers to global numbering, i.e. before mapping
      int ReadBackCounters(mappedFEB & FEB);
      void ReadBackAllCounters(){for(auto feb : febs) ReadBackCounters(feb);}
      int ResetCounters(mappedFEB & FEB);
      void ResetAllCounters(){for(auto feb : febs) ResetCounters(feb);}


      //Read datapath status values from FEB, store in subtree $odb_prefix/Variables/FEB datapath status
      //Parameter FPGA_ID refers to global numbering, i.e. before mapping
      int ReadBackDatapathStatus(mappedFEB & FEB);
      void ReadBackAllDatapathStatus(){for(auto feb : febs) ReadBackDatapathStatus(feb);}

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
      void setDummyConfig(mappedFEB & FEB,bool dummy = true);
  
      /**
       * use mutrig data emulator on fpga
       * n:    number of events per frame
       * fast: enable fast mode for data generator (shorter events)
       */
      void setDummyData_Enable(mappedFEB & FEB, bool dummy = true);
      void setDummyData_Count(mappedFEB & FEB, int n = 255);
      void setDummyData_Fast(mappedFEB & FEB, bool fast = false);
  
      /**
       * Disable data from specified ASIC (asic number in global numbering scheme, i.e. before mapping)
       */
      void setMask(int ASIC, bool value);
  
      /**
       * Disable PRBS decoder in FPGA
       */
      void setPRBSDecoderDisable(mappedFEB & FEB,bool disable);

      /**
       * Wait for lvds receivers ready strategy
       */
      void setWaitForAll(mappedFEB & FEB,bool val);
      void setWaitForAllSticky(mappedFEB & FEB,bool val);



      void syncReset(mappedFEB & FEB){chipReset(FEB);} //should be resetting the ASICs coarse counter only, missing pin on the asic. For future use
      void chipReset(mappedFEB & FEB); //reset all asics (digital part, CC, fsms, etc.)
      void DataPathReset(mappedFEB & FEB); //in FE-FPGA: everything upstream of merger (in the stream path)
      void LVDS_RX_Reset(mappedFEB & FEB); //in FE-FPGA: LVDS receiver blocks


      //reset signal alignment control
      void setResetSkewCphase(mappedFEB & FEB, BOOL cphase[]);
      void setResetSkewCdelay(mappedFEB & FEB, BOOL cdelay[]);
      void setResetSkewPhases(mappedFEB & FEB, INT phases[]);

};//class MutrigFEB

#endif // MUTRIG_FEB_H
