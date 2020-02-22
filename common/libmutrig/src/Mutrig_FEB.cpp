/********************************************************************\

  Name:         Mutrig_FEB.h
  Created by:   Konrad Briggl

Contents:       Definition of functions to talk to a mutrig-based FEB. Designed to no be derived into a SciFi_FEB and a Tiles_FEB class where subdetector specific actions are defined.
        Here: Definition of basic things for mutrig-configuration & datapath settings

\********************************************************************/

#include "Mutrig_FEB.h"
#include "midas.h"
#include "mfe.h" //for set_equipment_status

#include "mudaq_device_scifi.h"
#include "MutrigConfig.h"
#include "mutrig_midasodb.h"
#include <thread>
#include <chrono>
#include "reset_protocol.h"

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
    m_reg_shadow[FB_ID][FE_DPCTRL_REG]=0x1FFFFFFF;

    sprintf(set_str, "%s/Settings/Daq/dummy_config", m_odb_prefix);
    db_find_key(m_hDB, 0, set_str, &hTmp);
    assert(hTmp);
    on_settings_changed(m_hDB,hTmp,0,this);
    
    sprintf(set_str, "%s/Settings/Daq/dummy_data", m_odb_prefix);
    db_find_key(m_hDB, 0, set_str, &hTmp);
    assert(hTmp);
    on_settings_changed(m_hDB,hTmp,0,this);
    
    sprintf(set_str, "%s/Settings/daq/dummy_data_fast", m_odb_prefix);
    db_find_key(m_hDB, 0, set_str, &hTmp);
    assert(hTmp);
    on_settings_changed(m_hDB,hTmp,0,this);
    
    sprintf(set_str, "%s/Settings/Daq/dummy_data_n", m_odb_prefix);
    db_find_key(m_hDB, 0, set_str, &hTmp);
    assert(hTmp);
    on_settings_changed(m_hDB,hTmp,0,this);
    
    sprintf(set_str, "%s/Settings/Daq/prbs_decode_disable", m_odb_prefix);
    db_find_key(m_hDB, 0, set_str, &hTmp);
    assert(hTmp);
    on_settings_changed(m_hDB,hTmp,0,this);
    
    sprintf(set_str, "%s/Settings/Daq/LVDS_waitforall", m_odb_prefix);
    db_find_key(m_hDB, 0, set_str, &hTmp);
    assert(hTmp);
    on_settings_changed(m_hDB,hTmp,0,this);
    
    sprintf(set_str, "%s/Settings/Daq/LVDS_waitforall_sticky", m_odb_prefix);
    db_find_key(m_hDB, 0, set_str, &hTmp);
    assert(hTmp);
    on_settings_changed(m_hDB,hTmp,0,this);
    
    sprintf(set_str, "%s/Settings/Daq/mask", m_odb_prefix);
    db_find_key(m_hDB, 0, set_str, &hTmp);
    assert(hTmp);
    on_settings_changed(m_hDB,hTmp,0,this);
    
    sprintf(set_str, "%s/Settings/Daq/resetskew_cphase", m_odb_prefix);
    db_find_key(m_hDB, 0, set_str, &hTmp);
    assert(hTmp);
    on_settings_changed(m_hDB,hTmp,0,this);
    return 0;
}


int MutrigFEB::MapForEach(std::function<int(mutrig::MutrigConfig* /*mutrig config*/,int /*ASIC #*/)> func)
{
    INT status = DB_SUCCESS;
    //Iterate over ASICs
    for(unsigned int asic = 0; asic < GetNumASICs(); ++asic) {
        //ddprintf("mutrig_midasodb: Mapping %s, asic %d\n",prefix, asic);
        mutrig::MutrigConfig config(mutrig::midasODB::MapConfigFromDB(m_hDB,m_odb_prefix,asic));
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
       sprintf(path,"%s/Variables/Counters/nHits",m_odb_prefix);
       value=val[nASIC*15+0];
       odbval=value-last_counters[nASIC][0];
       if((status=db_set_value_index(m_hDB, 0, path, &odbval, val_size, nASIC, TID_DWORD, FALSE))!=DB_SUCCESS) return status;
//       printf("%s[%d]: %8.8x\n",path,nASIC,val[nASIC*15+0]);
       last_counters[nASIC][0]=value;

       sprintf(path,"%s/Variables/Counters/Time",m_odb_prefix);
       value=val[nASIC*15+2];
       odbval=value-last_counters[nASIC][1];
       if((status=db_set_value_index(m_hDB, 0, path, &odbval, val_size, nASIC, TID_DWORD, FALSE))!=DB_SUCCESS) return status;
//       printf("%s[%d]: %8.8x\n",path,nASIC,val[nASIC*15+2]);
       last_counters[nASIC][1]=value;

       sprintf(path,"%s/Variables/Counters/nBadFrames",m_odb_prefix);
       value=val[nASIC*15+3];
       odbval=value-last_counters[nASIC][2];
       if((status=db_set_value_index(m_hDB, 0, path, &odbval, val_size, nASIC, TID_DWORD, FALSE))!=DB_SUCCESS) return status;
//       printf("%s[%d]: %8.8x\n",path,nASIC,val[nASIC*15+3]);
       last_counters[nASIC][2]=value;

       sprintf(path,"%s/Variables/Counters/nFrames",m_odb_prefix);
       value=val[nASIC*15+5];
       odbval=value-last_counters[nASIC][3];
       if((status=db_set_value_index(m_hDB, 0, path, &odbval, val_size, nASIC, TID_DWORD, FALSE))!=DB_SUCCESS) return status;
//       printf("%s[%d]: %8.8x\n",path,nASIC,val[nASIC*15+5]);
       last_counters[nASIC][3]=value;

       sprintf(path,"%s/Variables/Counters/nErrorsPRBS",m_odb_prefix);
       value=val[nASIC*15+6];
       odbval=value-last_counters[nASIC][4];
       if((status=db_set_value_index(m_hDB, 0, path, &odbval, val_size, nASIC, TID_DWORD, FALSE))!=DB_SUCCESS) return status;
//       printf("%s[%d]: %8.8x\n",path,nASIC,val[nASIC*15+6]);
       last_counters[nASIC][4]=value;

       sprintf(path,"%s/Variables/Counters/nWordsPRBS",m_odb_prefix);
       value=val[nASIC*15+8];
       odbval=value-last_counters[nASIC][5];
       if((status=db_set_value_index(m_hDB, 0, path, &odbval, val_size, nASIC, TID_DWORD, FALSE))!=DB_SUCCESS) return status;
//       printf("%s[%d]: %8.8x\n",path,nASIC,val[nASIC*15+8]);
       last_counters[nASIC][5]=value;

       sprintf(path,"%s/Variables/Counters/nErrorsLVDS",m_odb_prefix);
       value=val[nASIC*15+9];
       odbval=value-last_counters[nASIC][6];
       if((status=db_set_value_index(m_hDB, 0, path, &odbval, val_size, nASIC, TID_DWORD, FALSE))!=DB_SUCCESS) return status;
//       printf("%s[%d]: %8.8x\n",path,nASIC,val[nASIC*15+9]);
       last_counters[nASIC][6]=value;

       sprintf(path,"%s/Variables/Counters/nWordsLVDS",m_odb_prefix);
       value=val[nASIC*15+11];
       odbval=value-last_counters[nASIC][7];
       if((status=db_set_value_index(m_hDB, 0, path, &odbval, val_size, nASIC, TID_DWORD, FALSE))!=DB_SUCCESS) return status;
//       printf("%s[%d]: %8.8x\n",path,nASIC,val[nASIC*15+11]);
       last_counters[nASIC][7]=value;

       sprintf(path,"%s/Variables/Counters/nDatasyncloss",m_odb_prefix);
       value=val[nASIC*15+12];
       odbval=value-last_counters[nASIC][8];
       if((status=db_set_value_index(m_hDB, 0, path, &odbval, val_size, nASIC, TID_DWORD, FALSE))!=DB_SUCCESS) return status;
//       printf("%s[%d]: %8.8x\n",path,nASIC,val[nASIC*15+14]);
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

   //printf("MutrigFEB::ReadBackDatapathStatus(): val[]={%8.8x,%8.8x,%8.8x} --> ",val[0],val[1],val[2]);
   char path[255];

   value=val[0] & (1<<0);
   sprintf(path, "%s/Variables/FEB datapath status/PLL locked", m_odb_prefix);
   if((status = db_set_value_index(m_hDB, 0, path, &value, sizeof(BOOL),FPGA_ID, TID_BOOL,false))!=DB_SUCCESS) return status;

   value=val[0] & (1<<8);
   sprintf(path, "%s/Variables/FEB datapath status/Buffer full", m_odb_prefix);
   if((status = db_set_value_index(m_hDB, 0, path, &value, sizeof(BOOL),FPGA_ID, TID_BOOL,false))!=DB_SUCCESS) return status;

   value=val[0] & (1<<4);
   sprintf(path, "%s/Variables/FEB datapath status/Frame desync", m_odb_prefix);
   if((status = db_set_value_index(m_hDB, 0, path, &value, sizeof(BOOL),FPGA_ID, TID_BOOL,false))!=DB_SUCCESS) return status;

   for(int i=0; i < nModulesPerFEB()*nAsicsPerModule(); i++){
      int a=FPGA_ID*nModulesPerFEB()*nAsicsPerModule()+i;
      sprintf(path, "%s/Variables/FEB datapath status/DPA locked", m_odb_prefix);
      value=(val[1]>>i) & 1;
      if((status = db_set_value_index(m_hDB, 0, path, &value, sizeof(BOOL),a, TID_BOOL,false))!=DB_SUCCESS) return status;

      sprintf(path, "%s/Variables/FEB datapath status/RX ready", m_odb_prefix);
      value=(val[2]>>i) & 1;
      if((status = db_set_value_index(m_hDB, 0, path, &value, sizeof(BOOL),a, TID_BOOL,false))!=DB_SUCCESS) return status;
   }

   return SUCCESS;
}





//MIDAS callback function for FEB register Setter functions
void MutrigFEB::on_settings_changed(HNDLE hDB, HNDLE hKey, INT, void * userdata)
{
   MutrigFEB* _this=static_cast<MutrigFEB*>(userdata);
   KEY key;
   db_get_key(hDB, hKey, &key);
   printf("MutrigFEB::on_settings_changed(%s)\n",key.name);
   INT ival;
   BOOL bval;
   INT bsize=sizeof(bval);
   INT isize=sizeof(ival);


   if (std::string(key.name) == "dummy_config") {
        db_get_data(hDB,hKey,&bval,&bsize,TID_BOOL);
        for(auto FEB: _this->m_FPGAs){
        if(!FEB.IsScEnabled()) continue; //skip disabled
        if(FEB.SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
        _this->setDummyConfig(FEB.SB_Port(),bval);
    }
   }
   if (std::string(key.name) == "dummy_data") {
        db_get_data(hDB,hKey,&bval,&bsize,TID_BOOL);
        for(auto FEB: _this->m_FPGAs){
        if(!FEB.IsScEnabled()) continue; //skip disabled
        if(FEB.SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
        _this->setDummyData_Enable(FEB.SB_Port(),bval);
    }

   }

   if (std::string(key.name) == "dummy_data_fast") {
        db_get_data(hDB,hKey,&bval,&bsize,TID_BOOL);
        for(auto FEB: _this->m_FPGAs){
        if(!FEB.IsScEnabled()) continue; //skip disabled
        if(FEB.SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
        _this->setDummyData_Fast(FEB.SB_Port(),bval);
    }
   }

   if (std::string(key.name) == "dummy_data_n") {
        db_get_data(hDB,hKey,&ival,&isize,TID_INT);
    for(auto FEB: _this->m_FPGAs){
        if(!FEB.IsScEnabled()) continue; //skip disabled
        if(FEB.SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
        _this->setDummyData_Count(FEB.SB_Port(),ival);
    }
   }

   if (std::string(key.name) == "prbs_decode_disable") {
        db_get_data(hDB,hKey,&bval,&bsize,TID_BOOL);
    for(auto FEB: _this->m_FPGAs){
        if(!FEB.IsScEnabled()) continue; //skip disabled
        if(FEB.SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
        _this->setPRBSDecoderDisable(FEB.SB_Port(),bval);
    }
   }

   if (std::string(key.name) == "LVDS_waitforall") {
        db_get_data(hDB,hKey,&bval,&bsize,TID_BOOL);
    for(auto FEB: _this->m_FPGAs){
        if(!FEB.IsScEnabled()) continue; //skip disabled
        if(FEB.SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
        _this->setWaitForAll(FEB.SB_Port(),bval);
    }
   }

   if (std::string(key.name) == "LVDS_waitforall_sticky") {
        db_get_data(hDB,hKey,&bval,&bsize,TID_BOOL);
	for(auto FEB: _this->m_FPGAs){
		if(!FEB.IsScEnabled()) continue; //skip disabled
		if(FEB.SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
		_this->setWaitForAllSticky(FEB.SB_Port(),bval);
	}
   }

   if (std::string(key.name) == "mask") {
      BOOL* barray=new BOOL[_this->GetNumASICs()];
      INT  barraysize=sizeof(BOOL)*_this->GetNumASICs();
      db_get_data(hDB,hKey,barray,&barraysize,TID_BOOL);
      
      for(int i=0;i<_this->GetNumASICs();i++){
         _this->setMask(i,barray[i]);
      }
      delete[] barray;
   }

   if (std::string(key.name) == "reset_datapath") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      if(value){
    for(auto FEB: _this->m_FPGAs){
        if(!FEB.IsScEnabled()) continue; //skip disabled
        if(FEB.SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
            _this->DataPathReset(FEB.SB_Port());
    }
        value = FALSE; // reset flag in ODB
        db_set_data(hDB, hKey, &value, sizeof(value), 1, TID_BOOL);
      }
   }
   if (std::string(key.name) == "reset_asics") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      if(value){
    for(auto FEB: _this->m_FPGAs){
        if(!FEB.IsScEnabled()) continue; //skip disabled
        if(FEB.SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
            _this->chipReset(FEB.SB_Port());
    }
        value = FALSE; // reset flag in ODB
        db_set_data(hDB, hKey, &value, sizeof(value), 1, TID_BOOL);
      }
   }
   if (std::string(key.name) == "reset_lvds") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      if(value){
         for(auto FEB: _this->m_FPGAs){
            if(!FEB.IsScEnabled()) continue; //skip disabled
            if(FEB.SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
            _this->LVDS_RX_Reset(FEB.SB_Port());
         }
         value = FALSE; // reset flag in ODB
         db_set_data(hDB, hKey, &value, sizeof(value), 1, TID_BOOL);
      }
   }

   //reset skew settings
   if ((std::string(key.name) == "resetskew_cphase")||
       (std::string(key.name) == "resetskew_cdelay")||
       (std::string(key.name) == "resetskew_phases")){
        char set_str[255];
        BOOL* cphase=new BOOL[_this->m_FPGAs.size()*_this->nModulesPerFEB()];
        BOOL* cdelay=new BOOL[_this->m_FPGAs.size()*_this->nModulesPerFEB()];
        INT  barraysize=sizeof(BOOL)*_this->m_FPGAs.size()*_this->nModulesPerFEB();
        INT*  phases=new INT[_this->m_FPGAs.size()*_this->nModulesPerFEB()];
        INT  iarraysize=sizeof(INT)*_this->m_FPGAs.size()*_this->nModulesPerFEB();
    
        sprintf(set_str, "%s/Settings/Daq/resetskew_cphase", _this->m_odb_prefix);
        db_find_key(hDB, 0, set_str, &hKey);
        assert(hKey);
        db_get_data(hDB,hKey,cphase,&barraysize,TID_BOOL);
        sprintf(set_str, "%s/Settings/Daq/resetskew_cdelay", _this->m_odb_prefix);
        db_find_key(hDB, 0, set_str, &hKey);
        assert(hKey);
        db_get_data(hDB,hKey,cdelay,&barraysize,TID_BOOL);
        sprintf(set_str, "%s/Settings/Daq/resetskew_phases", _this->m_odb_prefix);
        db_find_key(hDB, 0, set_str, &hKey);
        assert(hKey);
        db_get_data(hDB,hKey,phases,&iarraysize,TID_INT);
    
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
   if (std::string(key.name) == "reset_counters") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      if(value){
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
