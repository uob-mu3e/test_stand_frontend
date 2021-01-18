/********************************************************************\

  Name:         MuFEB.h
  Created by:   Konrad Briggl

Contents:       Definition of common functions to talk to a FEB. In particular common readout methods for status events and methods for slow control mapping are implemented here.

\********************************************************************/

#include "MuFEB.h"
#include "midas.h"
#include "odbxx.h"
#include "mfe.h" //for set_equipment_status

#include "mudaq_device_scifi.h"
#include "asic_config_base.h"
#include <thread>
#include <chrono>

//status flags from FEB
#define FEB_REPLY_SUCCESS 0
#define FEB_REPLY_ERROR   1

using midas::odb;

//handler function for update of switching board fiber mapping / status
void MuFEB::on_mapping_changed(HNDLE hDB, HNDLE hKey, INT, void * userdata)
{
   MuFEB* _this=static_cast<MuFEB*>(userdata);
   KEY key;
   db_get_key(hDB, hKey, &key);
   printf("MuFEB::on_mapping_changed(%s)\n",key.name);
   _this->RebuildFEBsMap();
}

void MuFEB::RebuildFEBsMap(){

    //clear map, we will rebuild it now
    m_FPGAs.clear();
    /*
        const char *link_settings_str[] = {
            "SwitchingBoardMask = INT[4] :",
            "SwitchingBoardNames = STRING[4] :",
            "FrontEndBoardMask = INT[192] :",
            "FrontEndBoardType = INT[192] :",
            "FrontEndBoardNames = STRING[192] :",
    */
    //TODO: create struct and use db_get_record

    //get FEB type -> find ours
    
    // get odb instance for links settings
    odb links_settings("/Equipment/Links/Settings");

    //fields to assemble fiber-driven name
    auto febtype = links_settings["FrontEndBoardType"];
    auto linkmask = links_settings["LinkMask"];
    auto sbnames = links_settings["SwitchingBoardNames"];
    auto febnames = links_settings["FrontEndBoardNames"];
    
    //fill our list. Currently only mapping primaries; secondary fibers for SciFi are implicitely mapped to the preceeding primary
    int lastPrimary=-1;
    int nSecondaries=0;
    char reportStr[255];
    for(uint16_t ID=0;ID<MAX_N_FRONTENDBOARDS;ID++){
        std::string name_link;
        std::string febnamesID;
        sbnames[ID/MAX_LINKS_PER_SWITCHINGBOARD].get(name_link);
        febnames[ID].get(febnamesID);
        name_link+=":";
        name_link+= febnamesID;
        if((INT) febtype[ID]==this->GetTypeID()){
            lastPrimary=m_FPGAs.size();
            m_FPGAs.push_back({ID,linkmask[ID],name_link.c_str()});
            sprintf(reportStr,"TX Fiber %d is mapped to Link %u \"%s\"                            --> SB=%u.%u %s", ID,m_FPGAs[lastPrimary].GetLinkID(),m_FPGAs[lastPrimary].GetLinkName().c_str(),
             m_FPGAs[lastPrimary].SB_Number(),m_FPGAs[lastPrimary].SB_Port(),
             !m_FPGAs[lastPrimary].IsScEnabled()?"\t[SC disabled]":"");
         cm_msg(MINFO,"MuFEB::RebuildFEBsMap",reportStr);
      }else if(IsSecondary(febtype[ID])){
     if(lastPrimary==-1){
            cm_msg(MERROR,"MuFEB::RebuildFEBsMap","Fiber #%d is set to type secondary but without primary",ID);
            return;
         }
         sprintf(reportStr,"TX Fiber %d is secondary, remap SC to primary ID %d, Link %u \"%s\" --> SB=%u.%u %s",
             ID,lastPrimary,m_FPGAs[lastPrimary].GetLinkID(),m_FPGAs[lastPrimary].GetLinkName().c_str(),
             m_FPGAs[lastPrimary].SB_Number(),m_FPGAs[lastPrimary].SB_Port(),
             !m_FPGAs[lastPrimary].IsScEnabled()?"\t[SC disabled]":"");
         cm_msg(MINFO,"MuFEB::RebuildFEBsMap",reportStr);
         lastPrimary=-1;
     nSecondaries++;
      }
   }
   sprintf(reportStr,"Found %lu FEBs of type %s, remapping %d secondaries.",m_FPGAs.size(),FEBTYPE_STR[GetTypeID()].c_str(),nSecondaries);
   cm_msg(MINFO,"MuFEB::RebuildFEBsMap",reportStr);

   //get SB mask -> update enable, overriding all FEB enables on that SB
   //TODO: is that actually needed? If SB is disabled, no SC for this one anyway!
   /*
   INT sbmask[MAX_N_SWITCHINGBOARDS];
   size = sizeof(INT)*MAX_N_SWITCHINGBOARDS;
   db_find_key(m_hDB, 0, "/Equipment/Links/Settings/SwitchingBoardMask", &hKey);
   assert(hKey);
   db_get_data(m_hDB, hKey, &sbmask, &size, TID_INT);
   for(size_t n=0;n<m_FPGAs.size();n++){
      assert(m_FPGAs[n].SB_Number()<MAX_N_SWITCHINGBOARDS);
      if(sbmask[m_FPGAs[n].SB_Number()]==0){
         m_FPGAs[n].mask=0;
      }
   }
   */
}

int MuFEB::WriteFEBID(){
    for(auto FEB: m_FPGAs){
       if(!FEB.IsScEnabled()) return SUCCESS; //skip disabled fibers
       if(FEB.SB_Number()!=m_SB_number) return SUCCESS; //skip commands not for this SB
       uint32_t val=0xFEB1FEB0;
       val+=(FEB.GetLinkID()<<16)+FEB.GetLinkID();

       char reportStr[255];
       sprintf(reportStr,"Setting FEBID of %s: Link%u, SB%u.%u to (%4.4x)-%4.4x",
             FEB.GetLinkName().c_str(),FEB.GetLinkID(),
             FEB.SB_Number(),FEB.SB_Port(),(val>>16)&0xffff,val&0xffff);
       cm_msg(MINFO,"MuFEB::WriteFEBID",reportStr);
       // ist the FF needed here? NB
       m_mu.FEBsc_write(FEB.SB_Port(), &val, 1 , (uint32_t) 0xFF00 | FPGA_ID_REGISTER_RW, m_ask_sc_reply);

    }

    return 0;

}

int MuFEB::ReadBackRunState(uint16_t FPGA_ID){
    //map to SB fiber
    auto FEB = m_FPGAs[FPGA_ID];

    //skip disabled fibers
    if(!FEB.IsScEnabled())
        return SUCCESS;

    //skip commands not for this SB
    if(FEB.SB_Number()!=m_SB_number)
        return SUCCESS;

   uint32_t val[2];
   int status=m_mu.FEBsc_read(FEB.SB_Port(), val, 2,0xFF00 | RUN_STATE_RESET_BYPASS_REGISTER_RW);
   if(status!=2) return status;
   //printf("MuFEB::ReadBackRunState(): val[]={%8.8x,%8.8x} --> %x,%x\n",val[0],val[1],val[0]&0x1ff,(val[0]>>16)&0x3ff);
   //val[0] is reset_bypass register
   //val[1] is reset_bypass payload
   char path[255];

   BOOL bypass_enabled=true;
   if(((val[0])&0x1ff)==0x000) bypass_enabled=false;
    // set odb value_index index = FPGA_ID, value = bypass_enabled
    sprintf(path, "%s/Variables/FEB Run State", m_odb_prefix);
    odb variables_feb_run_state(path);

    variables_feb_run_state["Bypass enabled&"][FPGA_ID] = bypass_enabled;

/*
// string variables are not possible with mlogger, so use raw state
   char state_str[32];
   switch((val[0]>>16)&0x3ff){
      case 1<<0:
      snprintf(state_str,32,"RUN_STATE_IDLE");
      break;
      case 1<<1:
      snprintf(state_str,32,"RUN_STATE_PREP");
      break;
      case 1<<2:
      snprintf(state_str,32,"RUN_STATE_SYNC");
      break;
      case 1<<3:
      snprintf(state_str,32,"RUN_STATE_RUNNING");
      break;
      case 1<<4:
      snprintf(state_str,32,"RUN_STATE_TERMINATING");
      break;
      case 1<<5:
      snprintf(state_str,32,"RUN_STATE_LINK_TEST");
      break;
      case 1<<6:
      snprintf(state_str,32,"RUN_STATE_SYNC_TEST");
      break;
      case 1<<7:
      snprintf(state_str,32,"RUN_STATE_RESET");
      break;
      case 1<<8:
      snprintf(state_str,32,"RUN_STATE_OUT_OF_DAQ");
      break;
      default:
      snprintf(state_str,32,"-broken-");
   }
   //printf("MuFEB::ReadBackRunState(): bypass=%s\n",bypass_enabled?"y":"n");
   //printf("MuFEB::ReadBackRunState(): current_state=%s\n",state_str);
*/
    // set odb value_index index = FPGA_ID, value = value
    DWORD value=(val[0]>>16) & 0x3ff;
    variables_feb_run_state["Run state&"][FPGA_ID] = value;

   return SUCCESS;
}


//Helper functions
uint32_t MuFEB::reg_setBit  (uint32_t reg_in, uint8_t bit, bool value){
    if(value)
        return (reg_in | 1<<bit);
    else
        return (reg_in & (~(1<<bit)));
}
uint32_t MuFEB::reg_unsetBit(uint32_t reg_in, uint8_t bit){return reg_setBit(reg_in,bit,false);}

bool MuFEB::reg_getBit(uint32_t reg_in, uint8_t bit){
    return (reg_in & (1<<bit)) != 0;
}

uint32_t MuFEB::reg_getRange(uint32_t reg_in, uint8_t length, uint8_t offset){
    return (reg_in>>offset) & ((1<<length)-1);
}
uint32_t MuFEB::reg_setRange(uint32_t reg_in, uint8_t length, uint8_t offset, uint32_t value){
    return (reg_in & ~(((1<<length)-1)<<offset)) | ((value & ((1<<length)-1))<<offset);
}


