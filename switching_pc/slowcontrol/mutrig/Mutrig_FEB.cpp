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


int MutrigFEB::WriteAll(uint32_t nasics){

    if ( nasics == 0 ) return 0;
    //initial Shadow register values

    //as a starting point, set all mask bits to 1. in the shadow register and override after.
    //This will ensure any asics that are not part of the detector configuration but exist in firmware are masked.
    for (size_t i = 0; i < febs.size(); i++) {
        m_reg_shadow[i][FE_DPCTRL_REG] = 0x1FFFFFFF;
    }

    // setup watches settings
    odb odb_set_str(odb_prefix+"/Settings/Daq");
    //use lambda function for passing this
    odb_set_str.watch([this](odb &o){
        on_settings_changed(o, this);
    });

    // setup watches commands
    odb odb_com_str(odb_prefix+"/Commands");
    //use lambda function for passing this
    odb_com_str.watch([this](odb &o){
        on_commands_changed(o, this);
    });

    return 0;
}


int MutrigFEB::MapForEach(std::function<int(mutrig::MutrigConfig* /*mutrig config*/,int /*ASIC #*/, int /*nModule #*/, int /*nASICperModule #*/)> func)
{
    INT status = DB_SUCCESS;
    // get asics from ODB
    odb odb_set_str(odb_prefix+"/Settings/Daq");
    odb settings_asics(odb_prefix + "/Settings/ASICs");
    uint32_t nasics = odb_set_str["num_asics"];
    uint32_t num_modules_per_feb = odb_set_str["num_modules_per_feb"];
    uint32_t num_asics_per_module = odb_set_str["num_asics_per_module"];

    //Iterate over ASICs
    for(unsigned int asic = 0; asic < nasics; ++asic) {
        //ddprintf("mutrig_midasodb: Mapping %s, asic %d\n",prefix, asic);
        mutrig::MutrigConfig config(mutrig::midasODB::MapConfigFromDB(settings_asics, asic));
        //note: this needs to be passed as pointer, otherwise there is a memory corruption after exiting the lambda
        status=func(&config, asic, num_modules_per_feb, num_asics_per_module);
        if (status != SUCCESS) break;
    }
    return status;
}


//ASIC configuration:
//Configure all asics under prefix (e.g. prefix="/Equipment/SciFi")
int MutrigFEB::ConfigureASICs(){
    cm_msg(MINFO, "setup_mutrig" , "Configuring MuTRiG asics under prefix %s/Settings/ASICs/", odb_prefix.c_str());
    int status = MapForEach([this](mutrig::MutrigConfig* config, int asic, int nModule, int nASICperModule){
            uint32_t rpc_status;

            // TODO: rework FEB mapping
            uint32_t FEB_ID = asic / (nASICperModule * nModule);
            if ( FEB_ID > febs.size() - 1 ) {
                printf(" [skipped -nofeb]\n");
                return FE_SUCCESS;
            }
            
            //mapping
            mappedFEB FEB = febs[FEB_ID];
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
                rpc_status = feb_sc.FEBsc_NiosRPC(FEB, feb::CMD_MUTRIG_ASIC_CFG | (asic % (nModule * nASICperModule)), payload);
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
    cm_msg(MINFO, "ConfigureASICsAllOff" , "Configuring all SciFi ASICs of nFEBs: %d in the ALL_OFF Mode", febs.size());
    int status = SUCCESS;
    for(size_t FPGA_ID = 0; FPGA_ID < febs.size(); FPGA_ID++){
        auto FEB = febs[FPGA_ID];
        if(!FEB.IsScEnabled())
            continue; //skip disabled fibers
        if(FEB.SB_Number()!= SB_number)
            continue; //skip commands not for this SB
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

    // rpc_ret is number of total ASICs n
    // Scifi Counters per ASIC N_ASICS_TOTAL
    // mutrig store:
    //  0: s_eventcounter
    //  1: s_timecounter low
    //  2: s_timecounter high
    //  3: s_crcerrorcounter
    //  4: s_framecounter
    //  5: s_prbs_wrd_cnt
    //  6: s_prbs_err_cnt
    // rx
    //  7: s_receivers_runcounter
    //  8: s_receivers_errorcounter
    //  9: s_receivers_synclosscounter
   
    // we have 10 counters per ASIC
    // read them back
    vector<uint32_t> val(rpc_ret*10);
    feb_sc.FEB_read(FEB, FEBSlowcontrolInterface::OFFSETS::FEBsc_RPC_DATAOFFSET, val);
    
    // get ASICs from ODB
    odb odb_set_str(odb_prefix+"/Settings/Daq");
    uint32_t num_asics_per_module = odb_set_str["num_asics_per_module"];
    uint32_t num_modules_per_feb = odb_set_str["num_modules_per_feb"];
    uint32_t num_asics_per_feb = num_asics_per_module * num_modules_per_feb;

   for(auto nASIC = 0 + FEB.SB_Port() * num_asics_per_feb; nASIC < num_asics_per_feb + FEB.SB_Port() * num_asics_per_feb; nASIC++){

        odb variables_counters(odb_prefix + "/Variables/Counters");

        // db_set_value_index
        variables_counters["nHits"][nASIC] = val[nASIC * 10 + 0];
        // For time we always get the first lower bits of time 1
        variables_counters["Time"][nASIC] = val[1];
        variables_counters["nBadFrames"][nASIC] = val[nASIC * 10 + 3];
        variables_counters["nFrames"][nASIC] = val[nASIC * 10 + 4];
        variables_counters["nErrorsPRBS"][nASIC] = val[nASIC * 10 + 6];
        variables_counters["nWordsPRBS"][nASIC] = val[nASIC * 10 + 5];
        variables_counters["nErrorsLVDS"][nASIC] = val[nASIC * 10 + 8];
        variables_counters["nWordsLVDS"][nASIC] = val[nASIC * 10 + 7];
        variables_counters["nDatasyncloss"][nASIC] = val[nASIC * 10 + 9];
   }
   // get rate per channel in counters
   vector<uint32_t> ch_rate(128);
   feb_sc.FEB_read(FEB, SCIFI_CH_RATE_REGISTER_R, ch_rate);
   for(size_t i_ch = 0; i_ch < ch_rate.size(); i_ch++){
        odb variables_counters(odb_prefix + "/Variables/Counters");
        variables_counters["Rate"][i_ch] = ch_rate[i_ch];
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


// MIDAS callback function commands changed
void MutrigFEB::on_commands_changed(odb o, void * userdata)
{
    std::string name = o.get_name();
    bool value = o;

    if (value)
        cm_msg(MINFO, "MutrigFEB::on_commands_changed", "Setting changed (%s)", name.c_str());

    MutrigFEB* _this=static_cast<MutrigFEB*>(userdata);

    if (name == "SciFiConfig" && o) {
          int status=_this->ConfigureASICs();
          if(status!=SUCCESS){ 
              cm_msg(MERROR, "SciFiConfig" , "ASIC Configuration failed.");
         	//TODO: what to do? 
          }
       o = false;
       return;
    }
    if (name == "SciFiAllOff" && o) {
        int status=_this->ConfigureASICsAllOff();
        if(status!=SUCCESS){
            cm_msg(MERROR, "SciFiAllOff" , "ASIC all off configuration failed. Return value was %d, expected %d.", status, SUCCESS);
            //TODO: what to do?
        }
       o = false;
       return;
    }
    if (name == "SciFiTDCTest") {
          int status=_this->ChangeTDCTest(o);
          if(status!=SUCCESS){
              cm_msg(MERROR, "SciFiConfig" , "Changing SciFi test pulses failed");
          }
          return;
    }
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
