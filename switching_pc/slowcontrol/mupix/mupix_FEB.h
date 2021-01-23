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
      static MupixFEB* m_instance; //singleton instance pointer: only one instance of MupixFEB
      MupixFEB(const MupixFEB&)=delete;
      MupixFEB(FEBSlowcontrolInterface & feb_sc_,
               const vector<mappedFEB> & febs_,
               const char* equipment_name_,
               const char* odb_prefix_,
               const uint8_t SB_number_)
        :
       MuFEB(feb_sc_, febs_, equipment_name_, odb_prefix_, SB_number_){}

   public:
      static MupixFEB* Create(FEBSlowcontrolInterface & feb_sc_,
                                    const vector<mappedFEB> & febs_,
                                    const char* equipment_name_,
                                    const char* odb_prefix_,
                                    const uint8_t SB_number_)
            {

                cm_msg(MINFO, "MupixFEB", "MupixFEB::Create(%s) as %s", odb_prefix_, equipment_name_);
                if(!m_instance)
                    m_instance=new MupixFEB(feb_sc_, febs_, equipment_name_, odb_prefix_, SB_number_);
                return m_instance;
            };



      static MupixFEB* Instance(){return m_instance;}


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
      uint32_t ReadBackCounters(uint16_t FPGA_ID);
      uint32_t ReadBackHitsEnaRate(uint16_t FPGA_ID);
      uint32_t getNFPGAs(){
          return febs.size();
      }
      void ReadBackAllCounters(DWORD** pdata){
          for(size_t i=0;i<febs.size();i++){
              (*pdata)++;
              **pdata = (DWORD)ReadBackCounters(i);
          };
      }


      //MIDAS callback for all setters below. Made static and using the user data argument as "this" to ease binding to C-style midas-callbacks
      static void on_settings_changed(odb o, void * userdata);

};//class MupixFEB

#endif // MUPIX_FEB_H
