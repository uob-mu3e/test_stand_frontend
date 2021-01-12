/********************************************************************\

  Name:         Mupix_FEB.h
  Created by:   Konrad Briggl

Contents:       Definition of functions to talk to a mupix-based FEB.

\********************************************************************/

#ifndef MUPIX_FEB_H
#define MUPIX_FEB_H
#include <map>
#include "midas.h"
#include "mudaq_device.h"
#include "mupix_config.h"
#include "MuFEB.h"
#include "odbxx.h"
using midas::odb;

class MupixFEB  : public MuFEB{
   private:
      std::map<uint8_t,std::map<uint32_t,uint32_t> > m_reg_shadow; /*[FPGA_ID][reg]*/
      static MupixFEB* m_instance; //singleton instance pointer: only one instance of MupixFEB
      MupixFEB(const MupixFEB&)=delete;
      MupixFEB(mudaq::MudaqDevice& mu, const char* equipment_name, const char* odb_prefix):
        MuFEB(mu,equipment_name,odb_prefix)
        {
		RebuildFEBsMap();
        };

   public:
      static MupixFEB* Create(mudaq::MudaqDevice& mu, const char* equipment_name, const char* odb_prefix){printf("MupixFEB::Create(%s) as %s\n",odb_prefix,equipment_name);if(!m_instance) m_instance=new MupixFEB(mu,equipment_name,odb_prefix); return m_instance;};
      static MupixFEB* Instance(){return m_instance;}


      //Mapping from ASIC number to FPGA_ID and ASIC_ID
      virtual uint16_t FPGAid_from_ID(int asic);
      virtual uint16_t ASICid_from_ID(int asic);

      uint16_t GetNumASICs();
      virtual FEBTYPE  GetTypeID(){return FEBTYPE::Pixel;}

      //ASIC configuration:
      //Configure all asics under prefix (e.g. prefix="/Equipment/Mupix")
      int ConfigureASICs();
      //Configure all boards under prefix (e.g. prefix="/Equipment/Mupix")
      int ConfigureBoards();

      //FEB registers and functions
      uint32_t ReadBackCounters(uint16_t FPGA_ID);
      uint32_t ReadBackHitsEnaRate(uint16_t FPGA_ID);
      uint32_t ReadBackMergerRate(uint16_t FPGA_ID);
      uint32_t ReadBackResetPhase(uint16_t FPGA_ID);
      uint32_t ReadBackTXReset(uint16_t FPGA_ID);
      uint32_t getNFPGAs(){
          return m_FPGAs.size();
      }
      void ReadBackAllCounters(DWORD** pdata){
          for(size_t i=0;i<m_FPGAs.size();i++){
              (*pdata)++;
              **pdata = (DWORD)ReadBackCounters(i);
          };
      }


      //MIDAS callback for all setters below. Made static and using the user data argument as "this" to ease binding to C-style midas-callbacks
      static void on_settings_changed(odb o, void * userdata);

};//class MupixFEB

#endif // MUPIX_FEB_H
