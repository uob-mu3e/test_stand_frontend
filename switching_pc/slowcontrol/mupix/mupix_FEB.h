/********************************************************************\

  Name:         Mupix_FEB.h
  Created by:   Konrad Briggl

Contents:       Definition of functions to talk to a mupix-based FEB.

\********************************************************************/

#ifndef MUPIX_FEB_H
#define MUPIX_FEB_H
#include <map>
#include "midas.h"
#include "FEBSlowcontrolInterface.h"
#include "mupix_config.h"
#include "MuFEB.h"
#include "odbxx.h"
#include "mupix_registers.h"

using midas::odb;

class MupixFEB  : public MuFEB{
    private:
      std::map<uint8_t,std::map<uint32_t,uint32_t> > m_reg_shadow; /*[FPGA_ID][reg]*/
      MupixFEB(const MupixFEB&)=delete;
      std::string odb_prefix;
      std::string pixel_odb_prefix;
    public:
      MupixFEB(FEBSlowcontrolInterface & feb_sc_,
               const vector<mappedFEB> & febs_,
               const uint64_t & febmask_,
               std::string switch_equipment_name_,
               std::string link_equipment_name_,
               std::string pixel_equipment_name_,
               const uint8_t SB_number_)
        :
       MuFEB(feb_sc_, febs_, febmask_, switch_equipment_name_, link_equipment_name_, SB_number_), 
              odb_prefix("/Equipment/" + switch_equipment_name_), 
              pixel_odb_prefix("/Equipment/" + pixel_equipment_name_)
              {}

      //Mapping from ASIC number to FPGA_ID and ASIC_ID
      virtual uint16_t FPGAid_from_ID(int asic) const;
      virtual uint16_t ASICid_from_ID(int asic) const;
      // TODO: Do this right
      virtual uint16_t GetModulesPerFEB() const {return 4;}
      virtual uint16_t GetASICSPerModule() const {return 3;}

      uint16_t GetNumASICs() const;
      virtual FEBTYPE  GetTypeID() const {return FEBTYPE::Pixel;}

      uint16_t ASICsPerFEB() const;

      //ASIC configuration:
      //Configure all asics under prefix (e.g. prefix="/Equipment/Mupix")
      int ConfigureASICs();
      int ConfigureTDACs();

      //FEB registers and functions
      uint32_t ReadBackLVDSNumHits(mappedFEB & FEB, uint16_t LVDS_ID);
      DWORD* ReadLVDSCounters(DWORD* pdata, mappedFEB & FEB);
      uint32_t ReadBackLVDSStatus(mappedFEB & FEB, uint16_t LVDS_ID);
      DWORD * ReadLVDSforPSLS(DWORD* pdata, mappedFEB & FEB);
  
      
      DWORD* fill_PSLL(DWORD* pdata){
          if ( febs.size() == 0 ) {
            // if no febs then send empty bank
            return pdata;
          }
          for(auto FEB : febs){
                pdata = ReadLVDSCounters(pdata, FEB);
          };
          return pdata;
      }

      DWORD* fill_PSLS(DWORD* pdata){
          if ( febs.size() == 0 ) {
            // if no febs then send empty bank
            return pdata;
          }
          for(auto FEB : febs){
                *pdata++ = FEB.GetLinkID();
                pdata = ReadLVDSforPSLS(pdata, FEB);
          };
          return pdata;
      }



      //MIDAS callback. Made static and using the user data argument as "this" to ease binding to C-style midas-callbacks
      static void on_settings_changed(odb o, void * userdata);

};//class MupixFEB

#endif // MUPIX_FEB_H
