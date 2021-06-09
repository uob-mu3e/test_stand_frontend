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

using midas::odb;

class MupixFEB  : public MuFEB{
   private:
      std::map<uint8_t,std::map<uint32_t,uint32_t> > m_reg_shadow; /*[FPGA_ID][reg]*/
      MupixFEB(const MupixFEB&)=delete;
    public:
      MupixFEB(FEBSlowcontrolInterface & feb_sc_,
               const vector<mappedFEB> & febs_,
               const uint64_t & febmask_,
               const char* equipment_name_,
               const char* odb_prefix_,
               const uint8_t SB_number_)
        :
       MuFEB(feb_sc_, febs_, febmask_, equipment_name_, odb_prefix_, SB_number_){}

      //Mapping from ASIC number to FPGA_ID and ASIC_ID
      virtual uint16_t FPGAid_from_ID(int asic) const;
      virtual uint16_t ASICid_from_ID(int asic) const;
      // TODO: Do this right
      virtual uint16_t GetModulesPerFEB() const {return 2;}
      virtual uint16_t GetASICSPerModule() const {return 4;}

      uint16_t GetNumASICs() const;
      virtual FEBTYPE  GetTypeID() const {return FEBTYPE::Pixel;}

      //ASIC configuration:
      //Configure all asics under prefix (e.g. prefix="/Equipment/Mupix")
      int ConfigureASICs();
      //Configure all boards under prefix (e.g. prefix="/Equipment/Mupix")
      int ConfigureBoards();

      //FEB registers and functions
      uint32_t ReadBackLVDSNumHits(uint16_t FPGA_ID, uint16_t LVDS_ID);
      uint32_t ReadBackLVDSNumHitsInMupixFormat(uint16_t FPGA_ID, uint16_t LVDS_ID);
      DWORD*  ReadLVDSCounters(DWORD* pdata, uint16_t FPGA_ID){
        for(uint32_t i=0; i<GetASICSPerModule()*GetModulesPerFEB(); i++){ // TODO: set currect LVDS links number
            // FPGA ID | Link ID
            *(DWORD*)pdata++ = (FPGA_ID << 16) | i;
            // number of hits from link
            *(DWORD*)pdata++ = ReadBackLVDSNumHits(FPGA_ID, i);
            // number of hits from link in mupix format
            *(DWORD*)pdata++ = ReadBackLVDSNumHitsInMupixFormat(FPGA_ID, i);
        };
        return pdata;
      };
      uint32_t getNFPGAs(){
          return febs.size();
      }
      // TODO: the febs.size() does not work, dont find out why thats why we pass the numFEBs
      DWORD* fill_PSLL(DWORD* pdata, uint32_t numFEBs){
          if ( numFEBs == 0 ) {
            // if no febs than send 3 zeros
            *(DWORD*)pdata++ = 0;
            *(DWORD*)pdata++ = 0;
            *(DWORD*)pdata++ = 0;
            return pdata;
          }
          for(uint16_t i=0; i<numFEBs; i++){
                pdata = ReadLVDSCounters(pdata, i);
          };
          return pdata;
      }


      //MIDAS callback for all setters below. Made static and using the user data argument as "this" to ease binding to C-style midas-callbacks
      static void on_settings_changed(odb o, void * userdata);

};//class MupixFEB

#endif // MUPIX_FEB_H
