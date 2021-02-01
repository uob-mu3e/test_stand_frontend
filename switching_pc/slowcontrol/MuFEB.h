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
#include "feblist.h"

class MuFEB {

   public:
      MuFEB(const MuFEB&)=delete;
      MuFEB(FEBSlowcontrolInterface & feb_sc_,
            const vector<mappedFEB> & febs_,
            const uint64_t & febmask_,
            const char* equipment_name_,
            const char* odb_prefix_,
            const uint8_t SB_number_):
              feb_sc(feb_sc_),
              febs(febs_),
              febmask(febmask_),
              equipment_name(equipment_name_),
              odb_prefix(odb_prefix_),
              SB_number(SB_number_)
        {}
      virtual ~MuFEB(){}

      const char* GetName(){return equipment_name;}
      const char* GetPrefix(){return odb_prefix;}

      virtual uint16_t GetNumASICs() const {return 0;}
      virtual uint16_t GetNumFPGAs() const {return febs.size();}
      virtual uint16_t GetModulesPerFEB() const {return 0;}
      virtual uint16_t GetASICSPerModule() const {return 0;}

      //Parameter FPGA_ID refers to global numbering, i.e. before mapping
      int ReadBackRunState(uint16_t FPGA_ID);
      void ReadBackAllRunState(){for(size_t i=0;i<febs.size();i++) ReadBackRunState(i);};

      int WriteFEBID();

      uint32_t ReadBackMergerRate(uint16_t FPGA_ID);
      uint32_t ReadBackResetPhase(uint16_t FPGA_ID);
      uint32_t ReadBackTXReset(uint16_t FPGA_ID);

      int fill_SSFE(DWORD * pdata);

protected:

      FEBSlowcontrolInterface & feb_sc;
      const vector<mappedFEB> & febs;
      const uint64_t & febmask;
      const char* equipment_name;
      const char* odb_prefix;
      const uint8_t SB_number;

      //Mapping from ASIC number to FPGA_ID and ASIC_ID
      virtual uint16_t FPGAid_from_ID(int asic) const {return 0;}; //global asic number to global FEB number
      virtual uint16_t ASICid_from_ID(int asic) const {return 0;}; //global asic number to FEB-local asic number

      //Return typeID for building FEB ID map
      virtual FEBTYPE GetTypeID() const {return 0;}
      virtual bool IsSecondary([[maybe_unused]] int t){return false;}

      //Helper functions
      uint32_t reg_setBit  (uint32_t reg_in, uint8_t bit, bool value=true);
      uint32_t reg_unsetBit(uint32_t reg_in, uint8_t bit);
      bool reg_getBit(uint32_t reg_in, uint8_t bit);
      uint32_t reg_getRange(uint32_t reg_in, uint8_t length, uint8_t offset);
      uint32_t reg_setRange(uint32_t reg_in, uint8_t length, uint8_t offset, uint32_t value);

};//class MuFEB


#endif // MUFEB_H
