/********************************************************************\

  Name:         SciFi_FEB.h
  Created by:   Konrad Briggl

Contents:       Class to alter settings on a SciFi-FE-FPGA. Derives from MutrigFEB, thus here only scifi-specific methods would be defined.

\********************************************************************/

#ifndef SCIFI_FEB_ACCESS_H
#define SCIFI_FEB_ACCESS_H

#include "midas.h"
#include "odbxx.h"
#include "FEBSlowcontrolInterface.h"
#include "Mutrig_FEB.h"

using midas::odb;

class SciFiFEB : public MutrigFEB{
   private:
      static SciFiFEB* m_instance; //signleton instance pointer: only one instance of SciFiFEB
      SciFiFEB(const SciFiFEB&)=delete;
      SciFiFEB(FEBSlowcontrolInterface & feb_sc_, const char* equipment_name, const char* odb_prefix)
	:
    MutrigFEB(feb_sc_, equipment_name, odb_prefix)
        {
		RebuildFEBsMap();
        };
   public:
      static SciFiFEB* Create(FEBSlowcontrolInterface & feb_sc_, const char* equipment_name, const char* odb_prefix)
      {
          
          cm_msg(MINFO, "SciFi_FEB", "SciFiFEB::Create(%s) as %s", odb_prefix, equipment_name);
          if(!m_instance)
              m_instance=new SciFiFEB(feb_sc_, equipment_name, odb_prefix);
          return m_instance;
      };
      static SciFiFEB* Instance(){return m_instance;};

      //Mapping from ASIC number to FPGA_ID and ASIC_ID
      virtual uint16_t FPGAid_from_ID(int asic);
      virtual uint16_t ASICid_from_ID(int asic);
      virtual uint16_t nModulesPerFEB(){return 2;}
      virtual uint16_t nAsicsPerModule(){return 4;}
      //Return typeID for building FEB ID map
      virtual FEBTYPE  GetTypeID(){return FEBTYPE::Fibre;}
      virtual bool IsSecondary(int t){return t==FEBTYPE::FibreSecondary;}


};//class SciFiFEB

#endif // SCIFI_FEB_ACCESS_H
