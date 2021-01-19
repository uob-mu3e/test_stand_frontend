/********************************************************************\

  Name:         MuFEB.h
  Created by:   Konrad Briggl

Contents:       Definition of common functions to talk to a FEB. In particular common readout methods for status events and methods for slow control mapping are implemented here.

\********************************************************************/

#include "MuFEB.h"
#include "midas.h"
#include "odbxx.h"
#include "mfe.h" //for set_equipment_status

#include "mudaq_device.h"
#include "asic_config_base.h"
#include <thread>
#include <chrono>

//status flags from FEB
#define FEB_REPLY_SUCCESS 0
#define FEB_REPLY_ERROR   1



//handler function for update of switching board fiber mapping / status
void MuFEB::on_mapping_changed(odb o, void * userdata)
{
   MuFEB* _this=static_cast<MuFEB*>(userdata);
   _this->RebuildFEBsMap();
}

void MuFEB::RebuildFEBsMap(){

    //clear map, we will rebuild it now
    m_FPGAs.clear();

    // get odb instance for links settings
    odb links_settings("/Equipment/Links/Settings");

    //fields to assemble fiber-driven name
    auto febtype = links_settings["FrontEndBoardType"];
    auto linkmask = links_settings["LinkMask"];
    auto sbnames = links_settings["SwitchingBoardNames"];
    auto febnames = links_settings["FrontEndBoardNames"];
    
    // fill our list. Currently only mapping primaries;
    // secondary fibers for SciFi are implicitely mapped to the preceeding primary
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
            cm_msg(MINFO,"MuFEB::RebuildFEBsMap","%s",reportStr);
        } else if(IsSecondary(febtype[ID])){
            if(lastPrimary==-1){
                cm_msg(MERROR,"MuFEB::RebuildFEBsMap","Fiber #%d is set to type secondary but without primary",ID);
                return;
            }
            sprintf(reportStr,"TX Fiber %d is secondary, remap SC to primary ID %d, Link %u \"%s\" --> SB=%u.%u %s",
                    ID,lastPrimary,m_FPGAs[lastPrimary].GetLinkID(),m_FPGAs[lastPrimary].GetLinkName().c_str(),
                    m_FPGAs[lastPrimary].SB_Number(),m_FPGAs[lastPrimary].SB_Port(),
                    !m_FPGAs[lastPrimary].IsScEnabled()?"\t[SC disabled]":"");
            cm_msg(MINFO,"MuFEB::RebuildFEBsMap","%s", reportStr);
            lastPrimary=-1;
            nSecondaries++;
        }
    }
    sprintf(reportStr,"Found %lu FEBs of type %s, remapping %d secondaries.",m_FPGAs.size(),FEBTYPE_STR[GetTypeID()].c_str(),nSecondaries);
    cm_msg(MINFO,"MuFEB::RebuildFEBsMap","%s", reportStr);
}

int MuFEB::WriteFEBID(){
    for(auto FEB: m_FPGAs){
       if(!FEB.IsScEnabled()) return SUCCESS; //skip disabled fibers
       if(FEB.SB_Number()!=m_SB_number) return SUCCESS; //skip commands not for this SB
       uint32_t val=0xFEB1FEB0; // TODO: Where does this hard-coded value come from?
       val+=(FEB.GetLinkID()<<16)+FEB.GetLinkID();

       char reportStr[255];
       sprintf(reportStr,"Setting FEBID of %s: Link%u, SB%u.%u to (%4.4x)-%4.4x",
             FEB.GetLinkName().c_str(),FEB.GetLinkID(),
             FEB.SB_Number(),FEB.SB_Port(),(val>>16)&0xffff,val&0xffff);
       cm_msg(MINFO,"MuFEB::WriteFEBID",reportStr);
       // ist the FF needed here? NB
       feb_sc.FEB_write(FEB.SB_Port(),  (uint32_t) 0xFF00 | FPGA_ID_REGISTER_RW, vector<uint32_t>(1,val));
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

   vector<uint32_t> val(2);
   char set_str[255];
   int status = feb_sc.FEB_read(FEB.SB_Port(), 0xFF00 | RUN_STATE_RESET_BYPASS_REGISTER_RW , val);
   if(status!=FEB_slowcontrol::ERRCODES::OK) return status;

   //val[0] is reset_bypass register
   //val[1] is reset_bypass payload
   char path[255];

   BOOL bypass_enabled=true;
   if(((val[0])&0x1ff)==0x000) bypass_enabled=false;
    // set odb value_index index = FPGA_ID, value = bypass_enabled
        
    sprintf(path, "%s/Variables", m_odb_prefix);
    odb variables_feb_run_state((std::string)path);

    sprintf(set_str, "Bypass enabled %d", FPGA_ID);
    variables_feb_run_state[set_str] = bypass_enabled;

    // set odb value_index index = FPGA_ID, value = value
    DWORD value=(val[0]>>16) & 0x3ff;
    sprintf(set_str, "Run state %d", FPGA_ID);
    variables_feb_run_state[set_str] = value;

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


