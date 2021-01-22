/********************************************************************\

  Name:         MuFEB.h
  Created by:   Konrad Briggl

Contents:       Definition of common functions to talk to a FEB. In particular common readout methods for status events and methods for slow control mapping are implemented here.

\********************************************************************/

#ifndef MUFEB_H
#define MUFEB_H
#include "odbxx.h"
#include "FEBSlowcontrolInterface.h"
#include "link_constants.h"
#include "feb_constants.h"

using midas::odb;

class MuFEB {
    protected:
      FEBSlowcontrolInterface & feb_sc;
      bool m_ask_sc_reply;
      const char* m_odb_prefix;
      const char* m_equipment_name;
      uint8_t m_SB_number;

   public:
      MuFEB(const MuFEB&)=delete;
      MuFEB(FEBSlowcontrolInterface & feb_sc_, const char* equipment_name, const char* odb_prefix):
              feb_sc(feb_sc_),
	      m_ask_sc_reply(true),
	      m_odb_prefix(odb_prefix),
	      m_equipment_name(equipment_name),
	      m_SB_number(0xff)
	{};
      void SetSBnumber(uint8_t n){m_SB_number=n;}
      const char* GetName(){return m_equipment_name;}
      const char* GetPrefix(){return m_odb_prefix;}

      virtual uint16_t GetNumASICs()=0;
      virtual uint16_t GetNumFPGAs(){return m_FPGAs.size();}

      void SetAskSCReply(bool ask){m_ask_sc_reply=ask;};

      //MIDAS callback for changed mapping of FEB IDs. Will clear m_FPGAs and rebuild this vector.
      static void on_mapping_changed(odb o, void * userdata);
      void RebuildFEBsMap();

      //Parameter FPGA_ID refers to global numbering, i.e. before mapping
      int ReadBackRunState(uint16_t FPGA_ID);
      void ReadBackAllRunState(){for(size_t i=0;i<m_FPGAs.size();i++) ReadBackRunState(i);};

      int WriteFEBID();

      uint32_t ReadBackMergerRate(uint16_t FPGA_ID);
      uint32_t ReadBackResetPhase(uint16_t FPGA_ID);
      uint32_t ReadBackTXReset(uint16_t FPGA_ID);

protected:
      //Mapping from ASIC number to FPGA_ID and ASIC_ID
      virtual uint16_t FPGAid_from_ID(int asic)=0; //global asic number to global FEB number
      virtual uint16_t ASICid_from_ID(int asic)=0; //global asic number to FEB-local asic number

      //Return typeID for building FEB ID map
      virtual FEBTYPE  GetTypeID()=0;
      virtual bool IsSecondary([[maybe_unused]] int t){return false;}


      //list of all FPGAs mapped to this subdetector. Used for pushing common configurations to all FEBs
      //TODO: extend to map<ID, FPGA_ID_TYPE> with more information (name, etc. for reporting).
      //TODO: add possibility to have febs with different number of asics (relevant only for pixel)
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

      //Helper functions
      uint32_t reg_setBit  (uint32_t reg_in, uint8_t bit, bool value=true);
      uint32_t reg_unsetBit(uint32_t reg_in, uint8_t bit);
      bool reg_getBit(uint32_t reg_in, uint8_t bit);
      uint32_t reg_getRange(uint32_t reg_in, uint8_t length, uint8_t offset);
      uint32_t reg_setRange(uint32_t reg_in, uint8_t length, uint8_t offset, uint32_t value);

};//class MuFEB


#endif // MUFEB_H
