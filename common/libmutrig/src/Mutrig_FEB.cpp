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

#include "mudaq_device_scifi.h"
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
#define SC_REG_OFFSET 0xff60
#define FE_DPMON_STATUS_REG    (SC_REG_OFFSET+0x4)
#define FE_DPMON_DPALOCK_REG   (SC_REG_OFFSET+0x5)
#define FE_DPMON_RXRDY_REG     (SC_REG_OFFSET+0x6)
#define FE_DPMON_RESERVED_REG  (SC_REG_OFFSET+0x7)
#define FE_DUMMYCTRL_REG       (SC_REG_OFFSET+0x8)
#define FE_DPCTRL_REG          (SC_REG_OFFSET+0x9)
#define FE_SUBDET_RESET_REG    (SC_REG_OFFSET+0xa)
#define FE_RESETSKEW_GLOBALS_REG  (SC_REG_OFFSET+0xb)
#define FE_SPIDATA_ADDR		0

//status flags from FEB
#define FEB_REPLY_SUCCESS 0
#define FEB_REPLY_ERROR   1


int MutrigFEB::WriteAll(){
    HNDLE hTmp;
    char set_str[255];
    if(GetNumASICs()==0) return 0;
    //initial Shadow register values

    //as a starting point, set all mask bits to 1. in the shadow register and override after.
    //This will ensure any asics that are not part of the detector configuration but exist in firmware are masked.
    for (size_t i=0;i<m_FPGAs.size();i++) {
        m_reg_shadow[i][FE_DPCTRL_REG]=0x1FFFFFFF;
    }

    sprintf(set_str, "%s/Settings/Daq", m_odb_prefix);
    odb odb_set_str(set_str);
    odb_set_str.watch(on_settings_changed);

    return 0;
}


int MutrigFEB::MapForEach(std::function<int(mutrig::MutrigConfig* /*mutrig config*/,int /*ASIC #*/)> func)
{
    INT status = DB_SUCCESS;
    //Iterate over ASICs
    for(unsigned int asic = 0; asic < GetNumASICs(); ++asic) {
        //ddprintf("mutrig_midasodb: Mapping %s, asic %d\n",prefix, asic);
        mutrig::MutrigConfig config(mutrig::midasODB::MapConfigFromDB(m_odb_prefix,asic));
        //note: this needs to be passed as pointer, otherwise there is a memory corruption after exiting the lambda
        status=func(&config,asic);
        if (status != SUCCESS) break;
    }
    return status;
}


//ASIC configuration:
//Configure all asics under prefix (e.g. prefix="/Equipment/SciFi")
int MutrigFEB::ConfigureASICs(){
      cm_msg(MINFO, "setup_mutrig" , "Configuring MuTRiG asics under prefix %s/Settings/ASICs/", m_odb_prefix);
   int status = MapForEach([this](mutrig::MutrigConfig* config, int asic){
      uint32_t rpc_status;
      //mapping
      uint16_t SB_ID=m_FPGAs[FPGAid_from_ID(asic)].SB_Number();
      uint16_t SP_ID=m_FPGAs[FPGAid_from_ID(asic)].SB_Port();
      uint16_t FA_ID=ASICid_from_ID(asic);

      if(!m_FPGAs[FPGAid_from_ID(asic)].IsScEnabled()){
          printf(" [skipped -nonenable]\n");
          return FE_SUCCESS;
      }
      if(SB_ID!=m_SB_number){
          printf(" [skipped -SB]\n");
          return FE_SUCCESS;
      }
      //printf("\n");

      cm_msg(MINFO, "setup_mutrig" , "Configuring MuTRiG asic %s/Settings/ASICs/%i/: Mapped to FEB%u -> SB%u.%u  ASIC #%d", m_odb_prefix,asic,FPGAid_from_ID(asic),SB_ID,SP_ID,FA_ID);


      try {
         //Write ASIC number & Configuraton
     //rpc_status=m_mu.FEBsc_NiosRPC(FPGAid_from_ID(asic),0x0110,{{reinterpret_cast<uint32_t*>(&asic),1},{reinterpret_cast<uint32_t*>(config->bitpattern_w), config->length_32bits}});
     rpc_status=m_mu.FEBsc_NiosRPC(SP_ID,0x0110,{{reinterpret_cast<uint32_t*>(&asic),1},{reinterpret_cast<uint32_t*>(config->bitpattern_w), config->length_32bits}});
      } catch(std::exception& e) {
          cm_msg(MERROR, "setup_mutrig", "Communication error while configuring MuTRiG %d: %s", asic, e.what());
          set_equipment_status(m_equipment_name, "SB-FEB Communication error", "red");
          return FE_ERR_HW; //note: return of lambda function
      }
      if(rpc_status!=FEB_REPLY_SUCCESS){
         //configuration mismatch, report and break foreach-loop
         set_equipment_status(m_equipment_name,  "MuTRiG config failed", "red");
         cm_msg(MERROR, "setup_mutrig", "MuTRiG configuration error for ASIC %i", asic);
         return FE_ERR_HW;//note: return of lambda function
      }
      return FE_SUCCESS;//note: return of lambda function
   });//MapForEach
   return status; //status of foreach function, SUCCESS when no error.
}

int MutrigFEB::ResetCounters(uint16_t FPGA_ID){
   //map to SB fiber
   auto FEB = m_FPGAs[FPGA_ID];
   if(!FEB.IsScEnabled()) return SUCCESS; //skip disabled fibers
   if(FEB.SB_Number()!=m_SB_number) return SUCCESS; //skip commands not for this SB

   auto rpc_ret=m_mu.FEBsc_NiosRPC(FEB.SB_Port(),0x0106,{});
   return rpc_ret;
}

int MutrigFEB::ReadBackCounters(uint16_t FPGA_ID){
   //map to SB fiber
   auto FEB = m_FPGAs[FPGA_ID];
   if(!FEB.IsScEnabled()) return SUCCESS; //skip disabled fibers
   if(FEB.SB_Number()!=m_SB_number) return SUCCESS; //skip commands not for this SB

   auto rpc_ret=m_mu.FEBsc_NiosRPC(FEB.SB_Port(),0x0105,{});
   //retrieve results
   uint32_t* val=new uint32_t[rpc_ret*5*3]; //nASICs * 5 counterbanks * 3 words
   INT val_size = sizeof(DWORD);
//   printf("RPC return: %u\n",rpc_ret);
   m_mu.FEBsc_read(FEB.SB_Port(), val, rpc_ret*5*3 , (uint32_t) m_mu.FEBsc_RPC_DATAOFFSET);
//   printf("done reading:\n");
//   for(int i=0;i<rpc_ret*4*3;i++){
//      printf("%2.2d %2.2d: %8.8x (%u)\n",i/(3*4),(i/3)%4,val[i],val[i]);
//   }
   static std::array<std::array<uint32_t,9>,8> last_counters;
   //store in midas
   INT status;
   int index=0;
   char path[255];
   uint32_t value;
   uint32_t odbval;
   for(int nASIC=0;nASIC<rpc_ret;nASIC++){

       // get midas odb object
       sprintf(path,"%s/Variables/Counters",m_odb_prefix);
       odb variables_counters(path);

       // db_set_value_index
       value=val[nASIC*15+0];
       odbval=value-last_counters[nASIC][0];
       variables_counters["nHits&"][nASIC] = odbval;
       last_counters[nASIC][0]=value;

       value=val[nASIC*15+2];
       odbval=value-last_counters[nASIC][1];
       variables_counters["Time&"][nASIC] = odbval;
       last_counters[nASIC][1]=value;

       value=val[nASIC*15+3];
       odbval=value-last_counters[nASIC][2];
       variables_counters["nBadFrames&"][nASIC] = odbval;
       last_counters[nASIC][2]=value;

       value=val[nASIC*15+5];
       odbval=value-last_counters[nASIC][3];
       variables_counters["nFrames&"][nASIC] = odbval;
       last_counters[nASIC][3]=value;

       value=val[nASIC*15+6];
       odbval=value-last_counters[nASIC][4];
       variables_counters["nErrorsPRBS&"][nASIC] = odbval;
       last_counters[nASIC][4]=value;

       value=val[nASIC*15+8];
       odbval=value-last_counters[nASIC][5];
       variables_counters["nWordsPRBS&"][nASIC] = odbval;
       last_counters[nASIC][5]=value;

       value=val[nASIC*15+9];
       odbval=value-last_counters[nASIC][6];
       variables_counters["nErrorsLVDS&"][nASIC] = odbval;
       last_counters[nASIC][6]=value;

       value=val[nASIC*15+11];
       odbval=value-last_counters[nASIC][7];
       variables_counters["nWordsLVDS&"][nASIC] = odbval;
       last_counters[nASIC][7]=value;

       value=val[nASIC*15+12];
       odbval=value-last_counters[nASIC][8];
       variables_counters["nDatasyncloss&"][nASIC] = odbval;
       last_counters[nASIC][8]=value;

   }

   delete[] val;
   return SUCCESS;
}

int MutrigFEB::ReadBackDatapathStatus(uint16_t FPGA_ID){
   //map to SB fiber
   auto FEB = m_FPGAs[FPGA_ID];
   if(!FEB.IsScEnabled()) return SUCCESS; //skip disabled fibers
   if(FEB.SB_Number()!=m_SB_number) return SUCCESS; //skip commands not for this SB

   uint32_t val[3];
   BOOL value;
   int status=m_mu.FEBsc_read(FEB.SB_Port(), val, 3, FE_DPMON_STATUS_REG);
   if(status!=3) return status;

    // get odb object
    char path[255];
    sprintf(path, "%s/Variables/FEB datapath status", m_odb_prefix);
    odb variables_feb_datapath_status(path);

    // db_set_value_index
    value=val[0] & (1<<0);
    variables_feb_datapath_status["PLL locked&"][FPGA_ID] = value;

    value=val[0] & (1<<8);
    variables_feb_datapath_status["Buffer full&"][FPGA_ID] = value;

    value=val[0] & (1<<4);
    variables_feb_datapath_status["Frame desync&"][FPGA_ID] = value;

   for(int i=0; i < nModulesPerFEB()*nAsicsPerModule(); i++){
       int a=FPGA_ID*nModulesPerFEB()*nAsicsPerModule()+i;
       value=(val[1]>>i) & 1;
       variables_feb_datapath_status["DPA locked&"][a] = value;

       value=(val[2]>>i) & 1;
       variables_feb_datapath_status["RX ready&"][a] = value;
   }

   return SUCCESS;
}





// MIDAS callback function for FEB register Setter functions
void MutrigFEB::on_settings_changed(odb o)
{
    std::string name = o.get_name();

    cm_msg(MINFO, "MutrigFEB::on_settings_changed", "Setting changed (%s)", name.c_str());

    MutrigFEB* _this=static_cast<MutrigFEB*>(this);

    INT ival;
    BOOL bval;

    if (name == "dummy_config") {
        bval = o;
        for(auto FEB : _this->m_FPGAs){
            if(!FEB.IsScEnabled()) continue; //skip disabled
            if(FEB.SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
            _this->setDummyConfig(FEB.SB_Port(), bval);
        }
    }

    if (name == "dummy_data") {
        bval = o;
        for(auto FEB: _this->m_FPGAs){
            if(!FEB.IsScEnabled()) continue; //skip disabled
            if(FEB.SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
            _this->setDummyData_Enable(FEB.SB_Port(),bval);
        }
    }

    if (name == "dummy_data_fast") {
        bval = o;
        for(auto FEB: _this->m_FPGAs){
            if(!FEB.IsScEnabled()) continue; //skip disabled
            if(FEB.SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
            _this->setDummyData_Fast(FEB.SB_Port(),bval);
        }
    }

    if (name == "dummy_data_n") {
        ival = o;
        for(auto FEB: _this->m_FPGAs){
            if(!FEB.IsScEnabled()) continue; //skip disabled
            if(FEB.SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
            _this->setDummyData_Count(FEB.SB_Port(),ival);
        }
    }

    if (name == "prbs_decode_disable") {
        bval = o;
        for(auto FEB: _this->m_FPGAs){
            if(!FEB.IsScEnabled()) continue; //skip disabled
            if(FEB.SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
            _this->setPRBSDecoderDisable(FEB.SB_Port(),bval);
        }
    }

    if (name == "LVDS_waitforall") {
        bval = o;
        for(auto FEB: _this->m_FPGAs){
            if(!FEB.IsScEnabled()) continue; //skip disabled
            if(FEB.SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
            _this->setWaitForAll(FEB.SB_Port(),bval);
        }
    }

    if (name == "LVDS_waitforall_sticky") {
        bval = o;
        for(auto FEB: _this->m_FPGAs){
            if(!FEB.IsScEnabled()) continue; //skip disabled
            if(FEB.SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
            _this->setWaitForAllSticky(FEB.SB_Port(),bval);
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
        if(bval){
            for(auto FEB : _this->m_FPGAs);
            if(!FEB.IsScEnabled()) continue; //skip disabled
            if(FEB.SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
            _this->DataPathReset(FEB.SB_Port());
        }
        value = FALSE; // reset flag in ODB
        o = value;
    }

    if (name == "reset_asics") {
        bval = o;
        if(bval){
            for(auto FEB: _this->m_FPGAs){
                if(!FEB.IsScEnabled()) continue; //skip disabled
                if(FEB.SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
                _this->chipReset(FEB.SB_Port());
            }
            value = FALSE; // reset flag in ODB
            o = value;
        }
    }

    if (name == "reset_lvds") {
        bval = o;
        if(bval){
            for(auto FEB: _this->m_FPGAs){
                if(!FEB.IsScEnabled()) continue; //skip disabled
                if(FEB.SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
                _this->LVDS_RX_Reset(FEB.SB_Port());
            }
            value = FALSE; // reset flag in ODB
            o = value;
        }
    }

    //reset skew settings
    if ((name == "resetskew_cphase")||
        (name == "resetskew_cdelay")||
        (name == "resetskew_phases")){

        char set_str[255];
        BOOL* cphase=new BOOL[_this->m_FPGAs.size()*_this->nModulesPerFEB()];
        BOOL* cdelay=new BOOL[_this->m_FPGAs.size()*_this->nModulesPerFEB()];
        INT*  phases=new INT[_this->m_FPGAs.size()*_this->nModulesPerFEB()];

        sprintf(set_str, "%s/Settings/Daq/resetskew_cphase", _this->m_odb_prefix);
        odb odb_resetskew_cphase(set_str);
        cphase = odb_resetskew_cphase;

        sprintf(set_str, "%s/Settings/Daq/resetskew_cdelay", _this->m_odb_prefix);
        odb odb_resetskew_cdelay(set_str);
        cdelay = odb_resetskew_cdelay;

        sprintf(set_str, "%s/Settings/Daq/resetskew_phases", _this->m_odb_prefix);
        odb odb_resetskew_phases(set_str);
        phases = odb_resetskew_phases;

        for(size_t i=0;i<_this->m_FPGAs.size();i++){
            if(_this->m_FPGAs[i].IsScEnabled()==false) continue; //skip disabled
            if(_this->m_FPGAs[i].SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
            BOOL* vals=new BOOL[_this->nModulesPerFEB()];
            vals[0]=cphase[i]; vals[1]=cphase[i+1];
            _this->setResetSkewCphase(_this->m_FPGAs[i].SB_Port(),vals);
            vals[0]=cdelay[i]; vals[1]=cdelay[i+1];
            _this->setResetSkewCdelay(_this->m_FPGAs[i].SB_Port(),vals);
            delete [] vals;
            INT* ivals=new INT[_this->nModulesPerFEB()];
            ivals[0]=phases[i]; ivals[1]=phases[i+1];
            _this->setResetSkewPhases(_this->m_FPGAs[i].SB_Port(),ivals);
            delete [] ivals;
        }
    }

    if (name == "reset_counters") {
        bval = odb_set_str;
        if(bval){
            _this->ResetAllCounters();
            value = FALSE; // reset flag in ODB
            db_set_data(hDB, hKey, &value, sizeof(value), 1, TID_BOOL);
        }
    }
}

//MutrigFEB registers and functions

/**
* Use emulated mutric on fpga for config
*/
void MutrigFEB::setDummyConfig(uint16_t FPGA_ID, bool dummy){
    //printf("MutrigFEB::setDummyConfig(%d)=%d\n",FPGA_ID,dummy);
    uint32_t val;

    //TODO: shadowing should know about broadcast FPGA ID
    //TODO: implement pull from FPGA when shadow value is not stored
    //m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
    //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG readback=%8.8x\n",FPGA_ID,val);
    val=m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG];

    val=reg_setBit(val,0,dummy);
    //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG new=%8.8x\n",FPGA_ID,val);
    m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG, m_ask_sc_reply);
    m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG]=val;
}

/**
* use mutrig data emulator on fpga
* n:    number of events per frame
* fast: enable fast mode for data generator (shorter events)
*/

void MutrigFEB::setDummyData_Enable(uint16_t FPGA_ID, bool dummy)
{
    //printf("MutrigFEB::setDummyData_Enable(%d)=%d\n",FPGA_ID,dummy);
    uint32_t val;

    val=m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG];
    //m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
    //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG readback=%8.8x\n",FPGA_ID,val);

    val=reg_setBit(val,1,dummy);
    //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG new=%8.8x\n",FPGA_ID,val);
    m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG, m_ask_sc_reply);
    m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG]=val;
}

void MutrigFEB::setDummyData_Fast(uint16_t FPGA_ID, bool fast)
{
    //printf("MutrigFEB::setDummyData_Fast(%d)=%d\n",FPGA_ID,fast);
    uint32_t  val;
    val=m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG];
    //m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
    //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG readback=%8.8x\n",FPGA_ID,val);

    val=reg_setBit(val,2,fast);
    //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG new=%8.8x\n",FPGA_ID,val);
    m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG, m_ask_sc_reply);
    m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG]=val;
}

void MutrigFEB::setDummyData_Count(uint16_t FPGA_ID, int n)
{
    if(n > 255) n = 255;
    //printf("MutrigFEB::setDummyData_Count(%d)=%d\n",FPGA_ID,n);
    uint32_t  val;
    val=m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG];
    //m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
    //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG readback=%8.8x\n",FPGA_ID,val);
    val=reg_setRange(val, 9, 3, n);

    //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG new=%8.8x\n",FPGA_ID,val);
    m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG,m_ask_sc_reply);
    m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG]=val;
}

/**
* Disable data from specified ASIC
*/
void MutrigFEB::setMask(int asic, bool value){
    //mapping
    uint16_t SB_ID=m_FPGAs[FPGAid_from_ID(asic)].SB_Number();
    uint16_t SP_ID=m_FPGAs[FPGAid_from_ID(asic)].SB_Port();
    uint16_t FB_ID=FPGAid_from_ID(asic);
    uint16_t FA_ID=ASICid_from_ID(asic);
    printf("MutrigFEB::setMask(%d)=%d (Mapped to %u->%u:%u:%u)\n",asic,value,FB_ID,SB_ID,SP_ID,FA_ID);
    
    if(!m_FPGAs[FPGAid_from_ID(asic)].IsScEnabled()) 
        return;
    if(SB_ID!=m_SB_number)
        return;
    
    uint32_t val;
    val=m_reg_shadow[FB_ID][FE_DPCTRL_REG];
    //m_mu.FEBsc_read(FPGAid_from_ID(asic), &val, 1 , (uint32_t) FE_DPCTRL_REG);
    //printf("MutrigFEB(%d)::FE_DPCTRL_REG readback=%8.8x\n",FPGAid_from_ID(asic),val);
    
    val=reg_setBit(val,FA_ID,value);
    //printf("MutrigFEB(%d)::FE_DPCTRL_REG new=%8.8x\n",FPGAid_from_ID(asic),val);
    m_mu.FEBsc_write(SP_ID, &val, 1 , (uint32_t) FE_DPCTRL_REG,m_ask_sc_reply);
    m_reg_shadow[FB_ID][FE_DPCTRL_REG]=val;
}



/**
* Disable prbs decoder in FPGA
*/
void MutrigFEB::setPRBSDecoderDisable(uint32_t FPGA_ID, bool disable){
    //printf("MutrigFEB::setPRBSDecoderDisable(%d)=%d\n",FPGA_ID,disable);
    uint32_t val;
    val=m_reg_shadow[FPGA_ID][FE_DPCTRL_REG];
    //m_mu.FEBsc_read(FPGAid_from_ID(asic), &val, 1 , (uint32_t) FE_DPCTRL_REG);
    //printf("MutrigFEB(%d)::FE_DPCTRL_REG readback=%8.8x\n",FPGAid_from_ID(asic),val);
    
    val=reg_setBit(val,31,disable);
    //printf("MutrigFEB(%d)::FE_DPCTRL_REG new=%8.8x\n",FPGA_ID,val);
    m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DPCTRL_REG,m_ask_sc_reply);
    m_reg_shadow[FPGA_ID][FE_DPCTRL_REG]=val;
}

void MutrigFEB::setWaitForAll(uint32_t FPGA_ID, bool value){
    //printf("MutrigFEB::setWaitForAll(%d)=%d\n",FPGA_ID,value);
    uint32_t val;
    val=m_reg_shadow[FPGA_ID][FE_DPCTRL_REG];
    //m_mu.FEBsc_read(FPGAid_from_ID(asic), &val, 1 , (uint32_t) FE_DPCTRL_REG);
    //printf("MutrigFEB(%d)::FE_DPCTRL_REG readback=%8.8x\n",FPGAid_from_ID(asic),val);
    
    val=reg_setBit(val,30,value);
    //printf("MutrigFEB(%d)::FE_DPCTRL_REG new=%8.8x\n",FPGAid_from_ID(asic),val);
    m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DPCTRL_REG,m_ask_sc_reply);
    m_reg_shadow[FPGA_ID][FE_DPCTRL_REG]=val;
}

void MutrigFEB::setWaitForAllSticky(uint32_t FPGA_ID, bool value){
    //printf("MutrigFEB::setWaitForAllSticky(%d)=%d\n",FPGA_ID,value);
    uint32_t val;
    val=m_reg_shadow[FPGA_ID][FE_DPCTRL_REG];
    //m_mu.FEBsc_read(FPGAid_from_ID(asic), &val, 1 , (uint32_t) FE_DPCTRL_REG);
    //printf("MutrigFEB(%d)::FE_DPCTRL_REG readback=%8.8x\n",FPGAid_from_ID(asic),val);
    
    val=reg_setBit(val,29,value);
    //printf("MutrigFEB(%d)::FE_DPCTRL_REG new=%8.8x\n",FPGAid_from_ID(asic),val);
    m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DPCTRL_REG,m_ask_sc_reply);
    m_reg_shadow[FPGA_ID][FE_DPCTRL_REG]=val;
}



//reset all asics (digital part, CC, fsms, etc.)
void MutrigFEB::chipReset(uint16_t FPGA_ID){
    uint32_t val=0;
    //m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG);
    //constant reset should not happen...
    //assert(!GET_FE_SUBDET_REST_BIT_CHIP(val));
    //set and clear reset
        val=reg_setBit(val,0,true);
    m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG,m_ask_sc_reply);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    val=reg_setBit(val,0,false);
    m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG,m_ask_sc_reply);
}

//reset full datapath upstream from merger
void MutrigFEB::DataPathReset(uint16_t FPGA_ID){
    uint32_t val=0;
    //m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG);
    //constant reset should not happen...
    //assert(!GET_FE_SUBDET_REST_BIT_DPATH(val));
    //set and clear reset
        val=reg_setBit(val,1,true);
    //do not expect a reply in write below, the data generator is in reset (not having sent a trailer) and this may block the data merger sending a slow control reply
    m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG,false);

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        val=reg_setBit(val,1,false);
    m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG,m_ask_sc_reply);
}

//reset lvds receivers
void MutrigFEB::LVDS_RX_Reset(uint16_t FPGA_ID){
    uint32_t val=0;
    //set and clear reset
    val=reg_setBit(val,2,true);
    m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG,m_ask_sc_reply);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    val=reg_setBit(val,2,false);
    m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_SUBDET_RESET_REG,m_ask_sc_reply);
}

//set reset skew configuration
void MutrigFEB::setResetSkewCphase(uint16_t FPGA_ID, BOOL cphase[]){
    uint32_t val=m_reg_shadow[FPGA_ID][FE_RESETSKEW_GLOBALS_REG];
    for(int i=0;i<nModulesPerFEB();i++){
        val=reg_setBit(val,i+6,cphase[i]);
    }
    m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_RESETSKEW_GLOBALS_REG, m_ask_sc_reply);
    m_reg_shadow[FPGA_ID][FE_RESETSKEW_GLOBALS_REG]=val;
}

void MutrigFEB::setResetSkewCdelay(uint16_t FPGA_ID, BOOL cdelay[]){
    uint32_t val=m_reg_shadow[FPGA_ID][FE_RESETSKEW_GLOBALS_REG];
    for(int i=0;i<nModulesPerFEB();i++){
        val=reg_setBit(val,i+10,cdelay[i]);
    }
    m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_RESETSKEW_GLOBALS_REG, m_ask_sc_reply);
    m_reg_shadow[FPGA_ID][FE_RESETSKEW_GLOBALS_REG]=val;
}

void MutrigFEB::setResetSkewPhases(uint16_t FPGA_ID, INT phases[]){
    uint32_t val[5];
    for(int i=0;i<nModulesPerFEB();i++){
        val[i]=phases[i];
    }
    m_mu.FEBsc_NiosRPC(FPGA_ID, 0x0104, {{val,nModulesPerFEB()}});
}
