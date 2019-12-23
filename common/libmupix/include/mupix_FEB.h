/********************************************************************\

  Name:         Mupix_FEB.h
  Created by:   Konrad Briggl

Contents:       Definition of functions to talk to a mupix-based FEB.

\********************************************************************/

#ifndef MUPIX_FEB_H
#define MUPIX_FEB_H
#include <map>
#include "midas.h"
#include "mudaq_device_scifi.h"
#include "mupix_config.h"

class MupixFEB {
   private:
      mudaq::MudaqDevice& m_mu;
      std::map<uint8_t,std::map<uint32_t,uint32_t> > m_reg_shadow; /*[FPGA_ID][reg]*/
      static MupixFEB* m_instance; //singleton instance pointer: only one instance of MupixFEB
      MupixFEB(const MupixFEB&)=delete;
      MupixFEB(mudaq::MudaqDevice& mu):m_mu(mu){;}

   public:
      static MupixFEB* Create(mudaq::MudaqDevice& mu){printf("FEB::Create()");if(!m_instance) m_instance=new MupixFEB(mu); return m_instance;}
      static MupixFEB* Instance(){return m_instance;}


      //Mapping from ASIC number to FPGA_ID and ASIC_ID
      static const uint8_t FPGA_broadcast_ID;
      virtual uint8_t FPGAid_from_ID(int asic);
      virtual uint8_t ASICid_from_ID(int asic);

      //ASIC configuration:
      //Configure all asics under prefix (e.g. prefix="/Equipment/Mupix")
      int ConfigureASICs(HNDLE hDB, const char* equipment_name, const char* odb_prefix);
      //Configure all boards under prefix (e.g. prefix="/Equipment/Mupix")
      int ConfigureBoards(HNDLE hDB, const char* equipment_name, const char* odb_prefix);

      //FEB registers and functions

      //MIDAS callback for all setters below. Made static and using the user data argument as "this" to ease binding to C-style midas-callbacks
      static void on_settings_changed(HNDLE hDB, HNDLE hKey, INT, void *);

};//class MupixFEB

#endif // MUPIX_FEB_H
