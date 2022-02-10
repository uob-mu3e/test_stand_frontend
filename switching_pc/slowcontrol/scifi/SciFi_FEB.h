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
   public:

      SciFiFEB(const SciFiFEB&)=delete;
      SciFiFEB(FEBSlowcontrolInterface & feb_sc_,
               const vector<mappedFEB> & febs_,
               const uint64_t & febmask_,
               std::string equipment_name_,
               std::string link_equipment_name_,
               const uint8_t SB_number_)
	:
    MutrigFEB(feb_sc_, febs_, febmask_, equipment_name_, link_equipment_name_, "/Equipment/"+equipment_name_, SB_number_){}


      //Mapping from ASIC number to FPGA_ID and ASIC_ID
      virtual uint16_t FPGAid_from_ID (int asic) const;
      virtual uint16_t ASICid_from_ID (int asic) const;
      virtual uint16_t GetModulesPerFEB() const {return 2;}
      virtual uint16_t GetASICSPerModule() const {return 4;}
      //Return typeID for building FEB ID map
      virtual FEBTYPE  GetTypeID() const {return FEBTYPE::Fibre;}
      virtual bool IsSecondary(int t){return t==FEBTYPE::FibreSecondary;}


};//class SciFiFEB

#endif // SCIFI_FEB_ACCESS_H
