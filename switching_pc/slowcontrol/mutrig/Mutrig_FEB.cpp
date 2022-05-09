/********************************************************************\

  Name:         Mutrig_FEB.h
  Created by:   Konrad Briggl

Contents:       Definition of functions to talk to a mutrig-based FEB. Designed to no be derived into a SciFi_FEB and a Tiles_FEB class where subdetector specific actions are defined.
        Here: Definition of basic things for mutrig-configuration & datapath settings

\********************************************************************/

#include "Mutrig_FEB.h"
#include "midas.h"
#include "odbxx.h"
#include "mfe.h" //for set_equipment_status

#include "../include/feb.h"
#include "scifi_registers.h"
using namespace mu3e::daq;

#include "MutrigConfig.h"
#include "mutrig_midasodb.h"
#include <thread>
#include <chrono>
#include "reset_protocol.h"

using midas::odb;

//offset for registers on nios SC memory
/*
 *            struct {
                struct {
                    alt_u32 ctrl;
                    alt_u32 nom;
                    alt_u64 denom;
                } counters;
                struct {
                    alt_u32 status;
                    alt_u32 rx_dpa_lock;
                    alt_u32 rx_ready;
                    alt_u32 reserved[1];
                } mon;
                struct {
                    alt_u32 dummy;
                    alt_u32 dp;
                    alt_u32 reset;
                    alt_u32 resetdelay;
                } ctrl;
            } scifi;

 * */

// TODO: Can we automatically sync this with the NIOS code?
// Should be possible like done with SCIFI_CTRL_DUMMY_REGISTER_W
#define SC_REG_OFFSET 0xff60
#define FE_DPMON_STATUS_REG    (SC_REG_OFFSET+0x4)
#define FE_DPMON_DPALOCK_REG   (SC_REG_OFFSET+0x5)
#define FE_DPMON_RXRDY_REG     (SC_REG_OFFSET+0x6)
#define FE_DPMON_RESERVED_REG  (SC_REG_OFFSET+0x7)
#define FE_DUMMYCTRL_REG       SCIFI_CTRL_DUMMY_REGISTER_W
#define FE_DPCTRL_REG          SCIFI_CTRL_DP_REGISTER_W
#define FE_SUBDET_RESET_REG    (SC_REG_OFFSET+0xa)
#define FE_RESETSKEW_GLOBALS_REG  (SC_REG_OFFSET+0xb)
#define FE_SPIDATA_ADDR		0

//status flags from FEB
#define FEB_REPLY_SUCCESS 0
#define FEB_REPLY_ERROR   1


int MutrigFEB::WriteAll(){

    if(GetNumASICs() == 0) return 0;
    //initial Shadow register values

    //as a starting point, set all mask bits to 1. in the shadow register and override after.
    //This will ensure any asics that are not part of the detector configuration but exist in firmware are masked.
    for (size_t i = 0; i < febs.size(); i++) {
        m_reg_shadow[i][FE_DPCTRL_REG] = 0x1FFFFFFF;
    }

    // setup watches
    odb odb_set_str(odb_prefix+"/Settings/Daq");
    odb_set_str.watch(on_settings_changed);

    return 0;
}


int MutrigFEB::MapForEach(std::function<int(mutrig::MutrigConfig* /*mutrig config*/,int /*ASIC #*/)> func)
{
    INT status = DB_SUCCESS;
    //Iterate over ASICs
    for(unsigned int asic = 0; asic < GetNumASICs(); ++asic) {
        //ddprintf("mutrig_midasodb: Mapping %s, asic %d\n",prefix, asic);
        mutrig::MutrigConfig config(mutrig::midasODB::MapConfigFromDB(odb_prefix,asic));
        //note: this needs to be passed as pointer, otherwise there is a memory corruption after exiting the lambda
        status=func(&config,asic);
        if (status != SUCCESS) break;
    }
    return status;
}


//ASIC configuration:
//Configure all asics under prefix (e.g. prefix="/Equipment/SciFi")
int MutrigFEB::ConfigureASICs(){
    cm_msg(MINFO, "setup_mutrig" , "Configuring MuTRiG asics under prefix %s/Settings/ASICs/", odb_prefix.c_str());
    int status = MapForEach([this](mutrig::MutrigConfig* config, int asic){
            uint32_t rpc_status;
            
            //mapping
            mappedFEB FEB = febs[FPGAid_from_ID(asic)];
            uint16_t SB_ID=FEB.SB_Number();
            uint16_t SP_ID=FEB.SB_Port();
            uint16_t FA_ID=ASICid_from_ID(asic);

            if(!FEB.IsScEnabled()){
                printf(" [skipped -nonenable]\n");
                return FE_SUCCESS;
            }
            if(SB_ID!= SB_number){
                printf(" [skipped -SB]\n");
                return FE_SUCCESS;
            }
            //printf("\n");

            cm_msg(MINFO,
                    "setup_mutrig" ,
                    "Configuring MuTRiG asic %s/Settings/ASICs/%i/: Mapped to FEB%u -> SB%u.%u  ASIC #%d",
                    odb_prefix.c_str(),asic,FPGAid_from_ID(asic),SB_ID,SP_ID,FA_ID);


            try {
                //Write ASIC number & Configuraton
                //ASIC number is the lowest byte of the command that's written to CMD_LEN_REGISTER_RW,
                //the configuration bitpattern is written to the RAM
                //rpc_status = m_mu.FEBsc_NiosRPC(FPGAid_from_ID(asic), feb::CMD_MUTRIG_ASIC_CFG, {{reinterpret_cast<uint32_t*>(&asic),1},{reinterpret_cast<uint32_t*>(config->bitpattern_w), config->length_32bits}});
                vector<vector<uint32_t>> payload;
                //printf("ASIC %d\n", asic);
                //uint32_t nb = 340;
                //do{
                //    nb--;
                //    printf("%02X ",reinterpret_cast<uint8_t*>(config->bitpattern_w)[nb]);
                //}while(nb>0);
                //printf("\n");

                payload.push_back(vector<uint32_t>(reinterpret_cast<uint32_t*>(config->bitpattern_w),reinterpret_cast<uint32_t*>(config->bitpattern_w)+config->length_32bits));
                // TODO: we make modulo number of asics per module here since each FEB has only # ASIC from 0 to asics per module but
                // here we loop until total number of asics which is asics per module times # of FEBs
                rpc_status = feb_sc.FEBsc_NiosRPC(FEB, feb::CMD_MUTRIG_ASIC_CFG | (asic % GetASICSPerFEB()), payload);
            } catch(std::exception& e) {
                cm_msg(MERROR, "setup_mutrig", "Communication error while configuring MuTRiG %d: %s", asic, e.what());
                set_equipment_status(equipment_name.c_str(), "SB-FEB Communication error", "red");
                return FE_ERR_HW; //note: return of lambda function
            }
            if(rpc_status!=FEB_REPLY_SUCCESS){
                //configuration mismatch, report and break foreach-loop
                set_equipment_status(equipment_name.c_str(),  "MuTRiG config failed", "red");
                cm_msg(MERROR, "setup_mutrig", "MuTRiG configuration error for ASIC %i", (asic % GetASICSPerFEB()));
                return FE_SUCCESS;//note: return of lambda function
            }
            return FE_SUCCESS;//note: return of lambda function
    });//MapForEach
    return status; //status of foreach function, SUCCESS when no error.
}

int MutrigFEB::ChangeTDCTest(bool o){
    cm_msg(MINFO, "SciFi ChangeTDCTest" , o ? "Turning on test pulses" : "Turning off test pulses");
    int status = feb_sc.ERRCODES::OK;
    for(size_t FPGA_ID = 0; FPGA_ID < febs.size(); FPGA_ID++){
        auto FEB = febs[FPGA_ID];
        if(!FEB.IsScEnabled())
            continue; //skip disabled fibers
        uint32_t regvalue;
        status = feb_sc.FEB_read(FEB, SCIFI_CNT_CTRL_REGISTER_W, regvalue);
        if(status != feb_sc.ERRCODES::OK) {
            cm_msg(MINFO, "SciFi ChangeTDCTest" , "Could not read control register");
            continue;
        }

        if(o)
            regvalue |= (1<<31);
        else
            regvalue &= ~(1<<31);

        status = feb_sc.FEB_write(FEB, SCIFI_CNT_CTRL_REGISTER_W, regvalue);
    }
    return status;
}
int MutrigFEB::ConfigureASICsAllOff(){
    cm_msg(MINFO, "ConfigureASICsAllOff" , "Configuring all SciFi ASICs in the ALL_OFF Mode");
    int status = SUCCESS;
    for(size_t FPGA_ID = 0; FPGA_ID < febs.size(); FPGA_ID++){
        auto FEB = febs[FPGA_ID];
        if(!FEB.IsScEnabled())
            continue; //skip disabled fibers
        auto rpc_ret = feb_sc.FEBsc_NiosRPC(FEB, feb::CMD_MUTRIG_ASIC_OFF, {});
        if (rpc_ret < 0){
            cm_msg(MERROR, "ConfigureASICsAllOff" , "Received negative return value from FEBsc_NiosRPC()");
            continue;
        }
    }
    return status;
}

int MutrigFEB::ResetCounters(mappedFEB & FEB){

    if(!FEB.IsScEnabled()) return SUCCESS; //skip disabled fibers
    if(FEB.SB_Number()!= SB_number) return SUCCESS; //skip commands not for this SB

    auto rpc_ret = feb_sc.FEBsc_NiosRPC(FEB, feb::CMD_MUTRIG_CNT_RESET, {});
    return rpc_ret;
}

int MutrigFEB::ReadBackCounters(mappedFEB & FEB){

   if(!FEB.IsScEnabled())
       return SUCCESS; //skip disabled fibers
   if(FEB.SB_Number()!= SB_number)
       return SUCCESS; //skip commands not for this SB

   auto rpc_ret = feb_sc.FEBsc_NiosRPC(FEB, feb::CMD_MUTRIG_CNT_READ, {});
   //retrieve results

   if(rpc_ret < 0) {
       cm_msg(MINFO, "ReadBackCounters", "RPC Returned %d for FPGA_ID %u", rpc_ret, FEB.SB_Port());
       return SUCCESS; // TODO: Proper error code
   }

   // rpc_ret is number of ASICs n, times 5 counters,
   // times (1 nominator + 2 denominator values)
   // The vector val is filled like
   // asic 1 counter 1 nominator
   // asic 1 counter 1 denominator upper 32 bit
   // asic 1 counter 1 denominator lower 32 bit
   // asic 1 counter 2 nominator
   // [...]
   // asic n counter 5 denominator lower 32 bit
   vector<uint32_t> val(rpc_ret*5*3);

   feb_sc.FEB_read(FEB, FEBSlowcontrolInterface::OFFSETS::FEBsc_RPC_DATAOFFSET,
                   val);

   static std::array<std::array<uint32_t,9>,8> last_counters;

   uint32_t value;
   uint32_t odbval;
   for(auto nASIC=0 + FEB.SB_Port() * GetASICSPerFEB(); nASIC < rpc_ret + FEB.SB_Port() * GetASICSPerFEB(); nASIC++){

 
       odb variables_counters(odb_prefix + "/Variables/Counters");

       // db_set_value_index
       value=val[nASIC*15+0];
       odbval=value-last_counters[nASIC][0];
       variables_counters["nHits"][nASIC] = odbval;
       last_counters[nASIC][0]=value;

       value=val[nASIC*15+2];
       odbval=value-last_counters[nASIC][1];
       variables_counters["Time"][nASIC] = odbval;
       last_counters[nASIC][1]=value;

       value=val[nASIC*15+3];
       odbval=value-last_counters[nASIC][2];
       variables_counters["nBadFrames"][nASIC] = odbval;
       last_counters[nASIC][2]=value;

       value=val[nASIC*15+5];
       odbval=value-last_counters[nASIC][3];
       variables_counters["nFrames"][nASIC] = odbval;
       last_counters[nASIC][3]=value;

       value=val[nASIC*15+6];
       odbval=value-last_counters[nASIC][4];
       variables_counters["nErrorsPRBS"][nASIC] = odbval;
       last_counters[nASIC][4]=value;

       value=val[nASIC*15+8];
       odbval=value-last_counters[nASIC][5];
       variables_counters["nWordsPRBS"][nASIC] = odbval;
       last_counters[nASIC][5]=value;

       value=val[nASIC*15+9];
       odbval=value-last_counters[nASIC][6];
       variables_counters["nErrorsLVDS"][nASIC] = odbval;
       last_counters[nASIC][6]=value;

       value=val[nASIC*15+11];
       odbval=value-last_counters[nASIC][7];
       variables_counters["nWordsLVDS"][nASIC] = odbval;
       last_counters[nASIC][7]=value;

       value=val[nASIC*15+12];
       odbval=value-last_counters[nASIC][8];
       variables_counters["nDatasyncloss"][nASIC] = odbval;
       last_counters[nASIC][8]=value;

   }

   return SUCCESS;
}

int MutrigFEB::ReadBackDatapathStatus(mappedFEB & FEB){

   if(!FEB.IsScEnabled()) return SUCCESS; //skip disabled fibers
   if(FEB.SB_Number()!= SB_number) return SUCCESS; //skip commands not for this SB

   vector<uint32_t> val(3);
   BOOL value;
   int status=feb_sc.FEB_read(FEB, FE_DPMON_STATUS_REG, val);
   if(status!=3) return status;

    odb variables_feb_datapath_status(odb_prefix + "/Variables/FEB datapath status");

    // db_set_value_index
    value=val[0] & (1<<0);
    variables_feb_datapath_status["PLL locked&"][FEB.SB_Port()] = value;

    value=val[0] & (1<<8);
    variables_feb_datapath_status["Buffer full&"][FEB.SB_Port()] = value;

    value=val[0] & (1<<4);
    variables_feb_datapath_status["Frame desync&"][FEB.SB_Port()] = value;

   for(int i=0; i < GetModulesPerFEB()*GetASICSPerModule(); i++){
       int a=FEB.SB_Port()*GetModulesPerFEB()*GetASICSPerModule()+i;
       value=(val[1]>>i) & 1;
       variables_feb_datapath_status["DPA locked&"][a] = value;

       value=(val[2]>>i) & 1;
       variables_feb_datapath_status["RX ready&"][a] = value;
   }

   return SUCCESS;
}





// MIDAS callback function for FEB register Setter functions
void MutrigFEB::on_settings_changed(odb o, void * userdata)
{
    std::string name = o.get_name();
    bool value = o;

    if (value)
        cm_msg(MINFO, "MutrigFEB::on_settings_changed", "Setting changed (%s)", name.c_str());

    MutrigFEB* _this=static_cast<MutrigFEB*>(userdata);

    INT ival;
    BOOL bval;

    if (name == "dummy_config") {
        bval = o;
        for(auto FEB : _this->febs){
            if(!FEB.IsScEnabled()) continue; //skip disabled
            if(FEB.SB_Number()!=_this->SB_number) continue; //skip commands not for me
            _this->setDummyConfig(FEB, bval);
        }
    }

    if (name == "dummy_data") {
        bval = o;
        for(auto FEB : _this->febs){
            if(!FEB.IsScEnabled()) continue; //skip disabled
            if(FEB.SB_Number()!=_this->SB_number) continue; //skip commands not for me
            _this->setDummyData_Enable(FEB,bval);
        }
    }

    if (name == "dummy_data_fast") {
        bval = o;
        for(auto FEB: _this->febs){
            if(!FEB.IsScEnabled()) continue; //skip disabled
            if(FEB.SB_Number()!=_this->SB_number) continue; //skip commands not for me
            _this->setDummyData_Fast(FEB,bval);
        }
    }

    if (name == "dummy_data_n") {
        ival = o;
        for(auto FEB : _this->febs){
            if(!FEB.IsScEnabled()) continue; //skip disabled
            if(FEB.SB_Number()!=_this->SB_number) continue; //skip commands not for me
            _this->setDummyData_Count(FEB,ival);
        }
    }

    if (name == "prbs_decode_disable") {
        bval = o;
        for(auto FEB : _this->febs){
            if(!FEB.IsScEnabled()) continue; //skip disabled
            if(FEB.SB_Number()!=_this->SB_number) continue; //skip commands not for me
            _this->setPRBSDecoderDisable(FEB,bval);
        }
    }

    if (name == "LVDS_waitforall") {
        bval = o;
        for(auto FEB : _this->febs){
            if(!FEB.IsScEnabled()) continue; //skip disabled
            if(FEB.SB_Number()!=_this->SB_number) continue; //skip commands not for me
            _this->setWaitForAll(FEB,bval);
        }
    }

    if (name == "LVDS_waitforall_sticky") {
        bval = o;
        for(auto FEB : _this->febs){
            if(!FEB.IsScEnabled()) continue; //skip disabled
            if(FEB.SB_Number()!=_this->SB_number) continue; //skip commands not for me
            _this->setWaitForAllSticky(FEB,bval);
        }
    }

    if (name == "mask") {
        //BOOL * barray=new BOOL[_this->GetNumASICs()];
        //barray = odb_set_str;
        for(int i=0;i<_this->GetNumASICs();i++){
            _this->setMask(i, o[i]);
        }
    }

    if (name == "reset_datapath") {
        bval = o;
        if(bval) {
            for (auto FEB : _this->febs) {
                if (!FEB.IsScEnabled()) continue; //skip disabled
                if (FEB.SB_Number() != _this->SB_number) continue; //skip commands not for me
                _this->DataPathReset(FEB);
            }
            o = FALSE; // reset flag in ODB
        }
    }
    

    if (name == "reset_asics") {
        bval = o;
        if(bval){
            for(auto FEB: _this->febs){
                if(!FEB.IsScEnabled()) continue; //skip disabled
                if(FEB.SB_Number()!=_this->SB_number) continue; //skip commands not for me
                _this->chipReset(FEB);
            }
            o = FALSE; // reset flag in ODB
        }
    }

    if (name == "reset_lvds") {
        bval = o;
        if(bval){
            for(auto FEB: _this->febs){
                if(!FEB.IsScEnabled()) continue; //skip disabled
                if(FEB.SB_Number()!=_this->SB_number) continue; //skip commands not for me
                _this->LVDS_RX_Reset(FEB);
            }
            o = FALSE; // reset flag in ODB
        }
    }

    //reset skew settings
    if ((name == "resetskew_cphase")||
        (name == "resetskew_cdelay")||
        (name == "resetskew_phases")){

        odb settings_daq(_this->odb_prefix + "/Settings/Daq");

        auto cphase = settings_daq["resetskew_cphase"];
        auto cdelay = settings_daq["resetskew_cdelay"];
        auto phases = settings_daq["resetskew_phases"];
        // TODO: Make sure indexing here is correct
        int index =0;
        for(auto FEB: _this->febs){
            int i = index++;
            if(FEB.IsScEnabled()==false) continue; //skip disabled
            if(FEB.SB_Number()!=_this->SB_number) continue; //skip commands not for me
            BOOL* vals=new BOOL[_this->GetModulesPerFEB()];
            vals[0]=cphase[i]; vals[1]=cphase[i+1];
            _this->setResetSkewCphase(FEB,vals);
            vals[0]=cdelay[i]; vals[1]=cdelay[i+1];
            _this->setResetSkewCdelay(FEB,vals);
            delete [] vals;
            INT* ivals=new INT[_this->GetModulesPerFEB()];
            ivals[0]=phases[i]; ivals[1]=phases[i+1];
            _this->setResetSkewPhases(FEB,ivals);
            delete [] ivals;
        }
    }

    if (name == "reset_counters") {
        bval = o;
        if(bval){
            _this->ResetAllCounters();
            o = FALSE; // reset flag in ODB
        }
    }
}

//MutrigFEB registers and functions

/**
* Use emulated mutric on fpga for config
*/
void MutrigFEB::setDummyConfig(mappedFEB & FEB, bool dummy){
    //printf("MutrigFEB::setDummyConfig(%d)=%d\n",FPGA_ID,dummy);
    uint32_t val;

    //TODO: shadowing should know about broadcast FPGA ID
    //TODO: implement pull from FPGA when shadow value is not stored
    //m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
    //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG readback=%8.8x\n",FPGA_ID,val);
    val=m_reg_shadow[FEB.SB_Port()][FE_DUMMYCTRL_REG];

    val=reg_setBit(val,0,dummy);
    //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG new=%8.8x\n",FPGA_ID,val);
    feb_sc.FEB_write(FEB, FE_DUMMYCTRL_REG, val);
    m_reg_shadow[FEB.SB_Port()][FE_DUMMYCTRL_REG]=val;
}

/**
* use mutrig data emulator on fpga
* n:    number of events per frame
* fast: enable fast mode for data generator (shorter events)
*/

void MutrigFEB::setDummyData_Enable(mappedFEB & FEB, bool dummy)
{
    //printf("MutrigFEB::setDummyData_Enable(%d)=%d\n",FPGA_ID,dummy);
    uint32_t val;

    val=m_reg_shadow[FEB.SB_Port()][FE_DUMMYCTRL_REG];
    //m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
    //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG readback=%8.8x\n",FPGA_ID,val);

    val=reg_setBit(val,1,dummy);
    //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG new=%8.8x\n",FPGA_ID,val);
    feb_sc.FEB_write(FEB, FE_DUMMYCTRL_REG, val);
    m_reg_shadow[FEB.SB_Port()][FE_DUMMYCTRL_REG]=val;
}

void MutrigFEB::setDummyData_Fast(mappedFEB & FEB, bool fast)
{
    //printf("MutrigFEB::setDummyData_Fast(%d)=%d\n",FPGA_ID,fast);
    uint32_t  val;
    val=m_reg_shadow[FEB.SB_Port()][FE_DUMMYCTRL_REG];
    //m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
    //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG readback=%8.8x\n",FPGA_ID,val);

    val=reg_setBit(val,2,fast);
    //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG new=%8.8x\n",FPGA_ID,val);
    feb_sc.FEB_write(FEB, FE_DUMMYCTRL_REG, val);
    m_reg_shadow[FEB.SB_Port()][FE_DUMMYCTRL_REG]=val;
}

void MutrigFEB::setDummyData_Count(mappedFEB & FEB, int n)
{
    if(n > 255) n = 255;
    //printf("MutrigFEB::setDummyData_Count(%d)=%d\n",FPGA_ID,n);
    uint32_t  val;
    val=m_reg_shadow[FEB.SB_Port()][FE_DUMMYCTRL_REG];
    //m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
    //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG readback=%8.8x\n",FPGA_ID,val);
    val=reg_setRange(val, 9, 3, n);

    //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG new=%8.8x\n",FPGA_ID,val);
    feb_sc.FEB_write(FEB, FE_DUMMYCTRL_REG, val);
    m_reg_shadow[FEB.SB_Port()][FE_DUMMYCTRL_REG]=val;
}

/**
* Disable data from specified ASIC
*/
void MutrigFEB::setMask(int asic, bool value){
    mappedFEB FEB = febs[FPGAid_from_ID(asic)];
    //mapping
    uint16_t SB_ID=FEB.SB_Number();
    uint16_t SP_ID=FEB.SB_Port();
    uint16_t FB_ID=FPGAid_from_ID(asic);
    uint16_t FA_ID=ASICid_from_ID(asic);
    printf("MutrigFEB::setMask(%d)=%d (Mapped to %u->%u:%u:%u)\n",asic,value,FB_ID,SB_ID,SP_ID,FA_ID);
    
    if(!febs[FPGAid_from_ID(asic)].IsScEnabled())
        return;
    if(SB_ID!=SB_number)
        return;
    
    uint32_t val;
    val=m_reg_shadow[FB_ID][FE_DPCTRL_REG];
    //m_mu.FEBsc_read(FPGAid_from_ID(asic), &val, 1 , (uint32_t) FE_DPCTRL_REG);
    //printf("MutrigFEB(%d)::FE_DPCTRL_REG readback=%8.8x\n",FPGAid_from_ID(asic),val);
    
    val=reg_setBit(val,FA_ID,value);
    //printf("MutrigFEB(%d)::FE_DPCTRL_REG new=%8.8x\n",FPGAid_from_ID(asic),val);
    feb_sc.FEB_write(FEB, FE_DPCTRL_REG, val);
    m_reg_shadow[FB_ID][FE_DPCTRL_REG]=val;
}



/**
* Disable prbs decoder in FPGA
*/
void MutrigFEB::setPRBSDecoderDisable(mappedFEB & FEB, bool disable){
    //printf("MutrigFEB::setPRBSDecoderDisable(%d)=%d\n",FPGA_ID,disable);
    uint32_t val;
    val=m_reg_shadow[FEB.SB_Port()][FE_DPCTRL_REG];
    //m_mu.FEBsc_read(FPGAid_from_ID(asic), &val, 1 , (uint32_t) FE_DPCTRL_REG);
    //printf("MutrigFEB(%d)::FE_DPCTRL_REG readback=%8.8x\n",FPGAid_from_ID(asic),val);
    
    val=reg_setBit(val,31,disable);
    //printf("MutrigFEB(%d)::FE_DPCTRL_REG new=%8.8x\n",FPGA_ID,val);
    feb_sc.FEB_write(FEB, FE_DPCTRL_REG, val);
    m_reg_shadow[FEB.SB_Port()][FE_DPCTRL_REG]=val;
}

void MutrigFEB::setWaitForAll(mappedFEB & FEB, bool value){
    //printf("MutrigFEB::setWaitForAll(%d)=%d\n",FPGA_ID,value);
    uint32_t val;
    val=m_reg_shadow[FEB.SB_Port()][FE_DPCTRL_REG];
    //m_mu.FEBsc_read(FPGAid_from_ID(asic), &val, 1 , (uint32_t) FE_DPCTRL_REG);
    //printf("MutrigFEB(%d)::FE_DPCTRL_REG readback=%8.8x\n",FPGAid_from_ID(asic),val);
    
    val=reg_setBit(val,30,value);
    //printf("MutrigFEB(%d)::FE_DPCTRL_REG new=%8.8x\n",FPGAid_from_ID(asic),val);
    feb_sc.FEB_write(FEB, FE_DPCTRL_REG, val);
    m_reg_shadow[FEB.SB_Port()][FE_DPCTRL_REG]=val;
}

void MutrigFEB::setWaitForAllSticky(mappedFEB & FEB, bool value){
    //printf("MutrigFEB::setWaitForAllSticky(%d)=%d\n",FPGA_ID,value);
    uint32_t val;
    val=m_reg_shadow[FEB.SB_Port()][FE_DPCTRL_REG];
    //m_mu.FEBsc_read(FPGAid_from_ID(asic), &val, 1 , (uint32_t) FE_DPCTRL_REG);
    //printf("MutrigFEB(%d)::FE_DPCTRL_REG readback=%8.8x\n",FPGAid_from_ID(asic),val);
    
    val=reg_setBit(val,29,value);
    //printf("MutrigFEB(%d)::FE_DPCTRL_REG new=%8.8x\n",FPGAid_from_ID(asic),val);
    feb_sc.FEB_write(FEB, FE_DPCTRL_REG, val);
    m_reg_shadow[FEB.SB_Port()][FE_DPCTRL_REG]=val;
}

// TODO: Collect in a single reset function
// TODO: Remove hardcoded bit patterns

//reset all asics (digital part, CC, fsms, etc.)
void MutrigFEB::chipReset(mappedFEB & FEB){
    uint32_t val=0;
    //m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) SCIFI_CTRL_RESET_REGISTER_W);
    //constant reset should not happen...
    //assert(!GET_FE_SUBDET_REST_BIT_CHIP(val));
    //set and clear reset
        val=reg_setBit(val,0,true);
    feb_sc.FEB_write(FEB, SCIFI_CTRL_RESET_REGISTER_W, val);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    val=reg_setBit(val,0,false);
    feb_sc.FEB_write(FEB, SCIFI_CTRL_RESET_REGISTER_W, val);
}

//reset full datapath upstream from merger
void MutrigFEB::DataPathReset(mappedFEB & FEB){
    uint32_t val=0;
    //m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) SCIFI_CTRL_RESET_REGISTER_W);
    //constant reset should not happen...
    //assert(!GET_FE_SUBDET_REST_BIT_DPATH(val));
    //set and clear reset
    val=reg_setBit(val,1,true);
    //do not expect a reply in write below, the data generator is in reset (not having sent a trailer) and this may block the data merger sending a slow control reply (TODO: this should be fixed in firmware!)
    feb_sc.FEB_write(FEB, SCIFI_CTRL_RESET_REGISTER_W, val);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    val=reg_setBit(val,1,false);
    feb_sc.FEB_write(FEB, SCIFI_CTRL_RESET_REGISTER_W, val);
}

//reset lvds receivers
void MutrigFEB::LVDS_RX_Reset(mappedFEB & FEB){
    uint32_t val=0;
    //set and clear reset
    val=reg_setBit(val,2,true);
    feb_sc.FEB_write(FEB, SCIFI_CTRL_RESET_REGISTER_W, val);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    val=reg_setBit(val,2,false);
    feb_sc.FEB_write(FEB, SCIFI_CTRL_RESET_REGISTER_W, val);
}

//set reset skew configuration
void MutrigFEB::setResetSkewCphase(mappedFEB & FEB, BOOL cphase[]){
    uint32_t val=m_reg_shadow[FEB.SB_Port()][FE_RESETSKEW_GLOBALS_REG];
    for(int i=0;i<GetModulesPerFEB();i++){
        val=reg_setBit(val,i+6,cphase[i]);
    }
    feb_sc.FEB_write(FEB, FE_RESETSKEW_GLOBALS_REG, val);
    m_reg_shadow[FEB.SB_Port()][FE_RESETSKEW_GLOBALS_REG]=val;
}

void MutrigFEB::setResetSkewCdelay(mappedFEB & FEB, BOOL cdelay[]){
    uint32_t val=m_reg_shadow[FEB.SB_Port()][FE_RESETSKEW_GLOBALS_REG];
    for(int i=0;i<GetModulesPerFEB();i++){
        val=reg_setBit(val,i+10,cdelay[i]);
    }
    feb_sc.FEB_write(FEB, FE_RESETSKEW_GLOBALS_REG, val);
    m_reg_shadow[FEB.SB_Port()][FE_RESETSKEW_GLOBALS_REG]=val;
}

void MutrigFEB::setResetSkewPhases(mappedFEB & FEB, INT phases[]){
    vector<uint32_t> val;
    for(int i=0;i<GetModulesPerFEB();i++){
        val.push_back(phases[i]);
    }
    feb_sc.FEBsc_NiosRPC(FEB, feb::CMD_MUTRIG_SKEW_RESET, vector<vector<uint32_t> >(1, val) );
}
