/********************************************************************\

  Name:         SciFi_FEB.h
  Created by:   Konrad Briggl

Contents:       Class to alter settings on a SciFi-FE-FPGA. Derives from MutrigFEB, thus here only scifi-specific methods would be defined.

\********************************************************************/

#ifndef SCIFI_FEB_ACCESS_H
#define SCIFI_FEB_ACCESS_H

#include "midas.h"
#include "mudaq_device_scifi.h"
#include "mutrig_config.h"
#include "Mutrig_FEB.h"
class SciFiFEB : public MutrigFEB{
   private:
      static SciFiFEB* m_instance; //signleton instance pointer: only one instance of SciFiFEB
      SciFiFEB(const SciFiFEB&)=delete;
      SciFiFEB(mudaq::MudaqDevice& mu):MutrigFEB(mu){};
   public:
      static SciFiFEB* Create(mudaq::MudaqDevice& mu){printf("FEB::Create()");if(!m_instance) m_instance=new SciFiFEB(mu); return m_instance;};
      static SciFiFEB* Instance(){return m_instance;};

      //Mapping from ASIC number to FPGA_ID and ASIC_ID
      virtual uint16_t FPGAid_from_ID(int asic);
      virtual uint16_t ASICid_from_ID(int asic);

      //MIDAS callback for all ___ SciFi specific ___ setters. Made static and using the user data argument as "this" to ease binding to C-style midas-callbacks
      static void on_scifi_settings_changed(HNDLE hDB, HNDLE hKey, INT, void *);

};//class SciFiFEB

#endif // SCIFI_FEB_ACCESS_H
