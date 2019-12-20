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
#include "MutrigConfig.h"
#include "link_constants.h"

class MutrigFEB {
   protected:
      mudaq::MudaqDevice& m_mu;
      std::map<uint16_t,std::map<uint32_t,uint32_t> > m_reg_shadow; /*[FPGA_ID][reg]*/
      bool m_ask_sc_reply;
      const char* m_odb_prefix;
      const char* m_equipment_name;
      uint8_t m_SB_number;

      HNDLE m_hDB;
   public:
      MutrigFEB(const MutrigFEB&)=delete;
      MutrigFEB(mudaq::MudaqDevice& mu, HNDLE hDB, const char* equipment_name, const char* odb_prefix):
	      m_mu(mu),
	      m_ask_sc_reply(true),
	      m_odb_prefix(odb_prefix),
	      m_equipment_name(equipment_name),
	      m_SB_number(0xff),
	      m_hDB(hDB)
	{};
      void SetSBnumber(uint8_t n){m_SB_number=n;}
      uint16_t GetNumASICs(){return m_FPGAs.size()*nModulesPerFEB()*nAsicsPerModule();}
      uint16_t GetNumModules(){return m_FPGAs.size()*nModulesPerFEB();}
      void SetAskSCReply(bool ask){m_ask_sc_reply=ask;};

      //MIDAS callback for all setters below (DAQ related, mapped to functions on FEB / settings from the DAQ subdirectory).
      //Made static and using the user data argument as "this" to ease binding to C-style midas-callbacks
      static void on_settings_changed(HNDLE hDB, HNDLE hKey, INT, void *);

      //MIDAS callback for changed mapping of FEB IDs. Will clear m_FPGAs and rebuild this vector.
      //Using user data argument as "this"
      //TODO: move to generic FEB class after merging with pixel SC
      static void on_mapping_changed(HNDLE hDB, HNDLE hKey, INT, void *);
      void RebuildFEBsMap();

      //Write all registers based on ODB values
      int WriteAll();

      //ASIC configuration:
      //Configure all asics under prefix (e.g. prefix="/Equipment/SciFi"), report any errors as equipment_name
      int ConfigureASICs();

      //Read counter values from FEB, store in subtree $odb_prefix/Variables/Counters/ 
      int ReadBackCounters(uint16_t FPGA_ID);

   protected:
      //Mapping from ASIC number to FPGA_ID and ASIC_ID
      virtual uint16_t FPGAid_from_ID(int asic)=0; //global asic number to global FEB number
      virtual uint16_t ASICid_from_ID(int asic)=0; //global asic number to FEB-local asic number
      virtual uint8_t nModulesPerFEB()=0;
      virtual uint8_t nAsicsPerModule()=0;
      //Return typeID for building FEB ID map
      virtual FEBTYPE  GetTypeID()=0;
      virtual bool IsSecondary(int t){return false;}

      //list of all FPGAs mapped to this subdetector. Used for pushing common configurations to all FEBs
      //TODO: move to generic FEB class after merging with pixel SC
      //TODO: extend to map<ID, FPGA_ID_TYPE> with more information (name, etc. for reporting).
      struct mapped_FEB_t{
	 private:
	 uint16_t LinkID;	//global numbering. sb_id=LinkID/MAX_LINKS_PER_SWITCHINGBOARD, sb_port=LinkID%MAX_LINKS_PER_SWITCHINGBOARD
	 INT mask; 
	 std::string fullname_link;
	 public:
	 mapped_FEB_t(uint16_t ID, INT linkmask, std::string physName):LinkID(ID),mask(linkmask),fullname_link(physName){};
	 bool IsScEnabled(){return mask&FEBLINKMASK::SCOn;}
	 bool IsDataEnabled(){return mask&FEBLINKMASK::DataOn;}
	 uint16_t GetLinkID(){return LinkID;}
	 std::string GetLinkName(){return fullname_link;}
	 //getters for FPGAPORT_ID and SB_ID (physical link address, independent on number of links per FEB)
	 uint8_t SB_Number(){return LinkID/MAX_LINKS_PER_SWITCHINGBOARD;}
	 uint8_t SB_Port()  {return LinkID%MAX_LINKS_PER_SWITCHINGBOARD;}
      };
      //map m_FPGAs[global_FEB_number] to a struct giving the physical link addres to a struct giving the physical link address
      std::vector<mapped_FEB_t> m_FPGAs;


      //Foreach loop over all asics under this prefix. Call with a lambda function,
      //e.g. midasODB::MapForEach(hDB, "/Equipment/SciFi",[mudaqdev_ptr](Config c,int asic){mudaqdev_ptr->ConfigureAsic(c,asic);});
      //Function must return SUCCESS, otherwise loop is stopped.
      int MapForEach(std::function<int(mutrig::MutrigConfig* /*mutrig config*/,int /*ASIC #*/)> func);





      //FEB registers and functions

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
       * Disable data from specified ASIC (asic number in global numbering scheme)
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



      void syncReset(uint16_t FPGA_ID){chipReset(FPGA_ID);}; //should be resetting the ASICs coarse counter only, missing pin on the asic. For future use
      void chipReset(uint16_t FPGA_ID); //reset all asics (digital part, CC, fsms, etc.)
      void DataPathReset(uint16_t FPGA_ID); //in FE-FPGA: everything upstream of merger (in the stream path)
      void LVDS_RX_Reset(uint16_t FPGA_ID); //in FE-FPGA: LVDS receiver blocks


      //reset signal alignment control
      void setResetSkewCphase(uint16_t FPGA_ID, BOOL cphase[]);
      void setResetSkewCdelay(uint16_t FPGA_ID, BOOL cdelay[]);
      void setResetSkewPhases(uint16_t FPGA_ID, INT phases[]);

};//class MutrigFEB

#endif // MUTRIG_FEB_H
