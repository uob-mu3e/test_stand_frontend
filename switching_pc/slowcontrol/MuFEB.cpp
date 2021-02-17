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

//status flags from FEB
#define FEB_REPLY_SUCCESS 0
#define FEB_REPLY_ERROR   1

using midas::odb;

int MuFEB::WriteFEBID(){
    for(auto FEB: febs){
       if(!FEB.IsScEnabled()) continue; //skip disabled fibers
       if(FEB.SB_Number()!=SB_number) continue; //skip commands not for this SB
       uint32_t val=0xFEB1FEB0; // TODO: Where does this hard-coded value come from?
       val+=(FEB.GetLinkID()<<16)+FEB.GetLinkID();

       char reportStr[255];
       sprintf(reportStr,"Setting FEBID of %s: Link%u, SB%u.%u to (%4.4x)-%4.4x",
             FEB.GetLinkName().c_str(),FEB.GetLinkID(),
             FEB.SB_Number(),FEB.SB_Port(),(val>>16)&0xffff,val&0xffff);
       cm_msg(MINFO,"MuFEB::WriteFEBID",reportStr);
       // ist the FF needed here? NB
       feb_sc.FEB_write(FEB.SB_Port(),  (uint32_t) 0xFF00 | FPGA_ID_REGISTER_RW, val);
    }
    return 0;
}

void MuFEB::ReadFirmwareVersionsToODB()
{
    vector<uint32_t> arria(1);
    vector<uint32_t> max(1);

    odb arriaversions("/Equipment/Switching/Variables/FEBFirmware/Arria V Firmware Version");
    odb maxversions("/Equipment/Switching/Variables/FEBFirmware/Max 10 Firmware Version");

    for(auto FEB: febs){
        if(!FEB.IsScEnabled()) continue; //skip disabled fibers
        if(FEB.SB_Number()!=SB_number) continue; //skip commands not for this SB

         if(!feb_sc.FEB_read(FEB.SB_Port(), GIT_HASH_REGISTER_R, arria) != FEBSlowcontrolInterface::ERRCODES::OK)
            cm_msg(MINFO,"MuFEB::ReadFirmwareVersionsToODB", "Failed to read Arria firmware version");
         else
            arriaversions[FEB.GetLinkID()] = arria[0];
         if(!feb_sc.FEB_read(FEB.SB_Port(), MAX10_VERSION_REGISTER_R, max) != FEBSlowcontrolInterface::ERRCODES::OK)
            cm_msg(MINFO,"MuFEB::ReadFirmwareVersionsToODB", "Failed to read Max firmware version");
         else
            maxversions[FEB.GetLinkID()] = max[0];
    }
}

int MuFEB::ReadBackRunState(uint16_t FPGA_ID){
    //map to SB fiber
    auto FEB = febs[FPGA_ID];

    //skip disabled fibers
    if(!FEB.IsScEnabled())
        return SUCCESS;

    //skip commands not for this SB
    if(FEB.SB_Number()!=SB_number)
        return SUCCESS;

   vector<uint32_t> val(2);
   char set_str[255];
   int status = feb_sc.FEB_read(FEB.SB_Port(), 0xFF00 | RUN_STATE_RESET_BYPASS_REGISTER_RW , val);
   if(status!=FEBSlowcontrolInterface::ERRCODES::OK) return status;

   //val[0] is reset_bypass register
   //val[1] is reset_bypass payload
   char path[255];

   BOOL bypass_enabled=true;
   if(((val[0])&0x1ff)==0x000) bypass_enabled=false;
    // set odb value_index index = FPGA_ID, value = bypass_enabled
        
    sprintf(path, "%s/Variables", odb_prefix);
    odb variables_feb_run_state((std::string)path);

    sprintf(set_str, "Bypass enabled %d", FPGA_ID);
    variables_feb_run_state[set_str] = bypass_enabled;

    // set odb value_index index = FPGA_ID, value = value
    DWORD value=(val[0]>>16) & 0x3ff;
    sprintf(set_str, "Run state %d", FPGA_ID);
    variables_feb_run_state[set_str] = value;

   return SUCCESS;
}

uint32_t MuFEB::ReadBackMergerRate(uint16_t FPGA_ID){
    auto FEB = febs[FPGA_ID];
    if(!FEB.IsScEnabled()) return SUCCESS; //skip disabled fibers
    if(FEB.SB_Number()!=SB_number) return SUCCESS; //skip commands not for this SB

    vector<uint32_t> mergerRate(1);
    feb_sc.FEB_read(FEB.SB_Port(), 0xFF00 | MERGER_RATE_REGISTER_R, mergerRate);
    return mergerRate[0];
}

uint32_t MuFEB::ReadBackResetPhase(uint16_t FPGA_ID){
    auto FEB = febs[FPGA_ID];
    if(!FEB.IsScEnabled()) return SUCCESS; //skip disabled fibers
    if(FEB.SB_Number()!=SB_number) return SUCCESS; //skip commands not for this SB

    vector<uint32_t> resetPhase(1);
    feb_sc.FEB_read(FEB.SB_Port(), 0xFF00 | RESET_PHASE_REGISTER_R, resetPhase);
    return resetPhase[0] & 0xFFFF;
}

uint32_t MuFEB::ReadBackTXReset(uint16_t FPGA_ID){
    auto FEB = febs[FPGA_ID];
    if(!FEB.IsScEnabled()) return SUCCESS; //skip disabled fibers
    if(FEB.SB_Number()!=SB_number) return SUCCESS; //skip commands not for this SB

    vector<uint32_t> TXReset(1);
    feb_sc.FEB_read(FEB.SB_Port(), 0xFF00 | RESET_OPTICAL_LINKS_REGISTER_RW, TXReset);
    return TXReset[0] & 0xFFFFFFFC;
}

int MuFEB::fill_SSFE(DWORD *pdata)
{
    uint32_t index = 0;

    for(auto FEB: febs){
       if(!FEB.IsScEnabled()) continue; //skip disabled fibers
       if(FEB.SB_Number()!=SB_number) continue;

       uint32_t port = FEB.SB_Port();
       // Fill in zeroes for non-existing ports
       while(index < port){
            // 26 is per_fe_SSFE_size - need to find a header for that...
            for(uint32_t j=0; j < 26; j++){
                *pdata++ = 0;
             }
           index++;
       }

       // And here we would actually fill the bank
       for(uint32_t j=0; j < 26; j++){
           *pdata++ = 0;
        }
        index++;
    }



    // Fill in zeroes for non-existing ports
    while(index < N_FEBS[SB_number]){
        for(uint32_t j=0; j < 26; j++){
            *pdata++ = 0;
         }
       index++;
    }

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


