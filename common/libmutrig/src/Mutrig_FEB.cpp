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
#define SC_REG_OFFSET 0xff60
#define FE_DUMMYCTRL_REG       (SC_REG_OFFSET+0x8)
#define FE_DPCTRL_REG          (SC_REG_OFFSET+0x9)
#define FE_SUBDET_RESET_REG    (SC_REG_OFFSET+0xa)
#define FE_RESETSKEW_GLOBALS_REG  (SC_REG_OFFSET+0xb)
#define FE_SPIDATA_ADDR		0

//status flags from FEB
#define FEB_REPLY_SUCCESS 0
#define FEB_REPLY_ERROR   1


//handler function for update of switching board fiber mapping / status
void MutrigFEB::on_mapping_changed(HNDLE hDB, HNDLE hKey, INT, void * userdata)
{
   MutrigFEB* _this=static_cast<MutrigFEB*>(userdata);
   KEY key;
   db_get_key(hDB, hKey, &key);
   printf("MutrigFEB::on_mapping_changed(%s)\n",key.name);
   _this->RebuildFEBsMap();
}

void MutrigFEB::RebuildFEBsMap(){
   HNDLE hKey;
   int size;

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
   INT febtype[MAX_N_FRONTENDBOARDS];
   size = sizeof(INT)*MAX_N_FRONTENDBOARDS;
   db_find_key(m_hDB, 0, "/Equipment/Links/Settings/FrontEndBoardType", &hKey);
   assert(hKey);
   db_get_data(m_hDB, hKey, &febtype, &size, TID_INT);
   //get FEB mask -> set enable for our FEBs
   INT linkmask[MAX_N_FRONTENDBOARDS];
   size = sizeof(INT)*MAX_N_FRONTENDBOARDS;
   db_find_key(m_hDB, 0, "/Equipment/Links/Settings/LinkMask", &hKey);
   assert(hKey);
   db_get_data(m_hDB, hKey, &linkmask, &size, TID_INT);
   //fields to assemble fiber-driven name
   char sbnames[MAX_N_SWITCHINGBOARDS][32];
   size = sizeof(char)*MAX_N_SWITCHINGBOARDS*32;
   db_find_key(m_hDB, 0, "/Equipment/Links/Settings/SwitchingBoardNames", &hKey);
   assert(hKey);
   db_get_data(m_hDB, hKey, &sbnames, &size, TID_STRING);
   char febnames[MAX_N_FRONTENDBOARDS][32];
   size = sizeof(char)*MAX_N_FRONTENDBOARDS*32;
   db_find_key(m_hDB, 0, "/Equipment/Links/Settings/FrontEndBoardNames", &hKey);
   assert(hKey);
   db_get_data(m_hDB, hKey, &febnames, &size, TID_STRING);

   //fill our list. Currently only mapping primaries; secondary fibers for SciFi are implicitely mapped to the preceeding primary
   int lastPrimary=-1;
   int nSecondaries=0;
   char reportStr[255];
   for(uint16_t ID=0;ID<MAX_N_FRONTENDBOARDS;ID++){
      std::string name_link;
      name_link=sbnames[ID/MAX_LINKS_PER_SWITCHINGBOARD];
      name_link+=":";
      name_link+=febnames[ID];
      if(febtype[ID]==this->GetTypeID()){
         lastPrimary=m_FPGAs.size();
         m_FPGAs.push_back({ID,linkmask[ID],name_link.c_str()});
         sprintf(reportStr,"TX Fiber %d is mapped to Link %u \"%s\"                            --> SB=%u.%u %s",
             ID,m_FPGAs[lastPrimary].GetLinkID(),m_FPGAs[lastPrimary].GetLinkName().c_str(),
             m_FPGAs[lastPrimary].SB_Number(),m_FPGAs[lastPrimary].SB_Port(),
             !m_FPGAs[lastPrimary].IsScEnabled()?"\t[SC disabled]":"");
         cm_msg(MINFO,"MutrigFEB::RebuildFEBsMap",reportStr);
      }else if(IsSecondary(febtype[ID])){
     if(lastPrimary==-1){
            cm_msg(MERROR,"MutrigFEB::RebuildFEBsMap","Fiber #%d is set to type secondary but without primary",ID);
            return;
         }
         sprintf(reportStr,"TX Fiber %d is secondary, remap SC to primary ID %d, Link %u \"%s\" --> SB=%u.%u %s",
             ID,lastPrimary,m_FPGAs[lastPrimary].GetLinkID(),m_FPGAs[lastPrimary].GetLinkName().c_str(),
             m_FPGAs[lastPrimary].SB_Number(),m_FPGAs[lastPrimary].SB_Port(),
             !m_FPGAs[lastPrimary].IsScEnabled()?"\t[SC disabled]":"");
         cm_msg(MINFO,"MutrigFEB::RebuildFEBsMap",reportStr);
         lastPrimary=-1;
     nSecondaries++;
      }
   }
   sprintf(reportStr,"Found %lu FEBs of type %s, remapping %d secondaries.",m_FPGAs.size(),FEBTYPE_STR[GetTypeID()].c_str(),nSecondaries);
   cm_msg(MINFO,"MutrigFEB::RebuildFEBsMap",reportStr);

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


int MutrigFEB::WriteAll(){
    HNDLE hTmp;
    char set_str[255];
    
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
   int status = MapForEach([this](mutrig::MutrigConfig* config, int asic){
      uint32_t rpc_status;
      //mapping
      uint16_t SB_ID=m_FPGAs[FPGAid_from_ID(asic)].SB_Number();
      uint16_t SP_ID=m_FPGAs[FPGAid_from_ID(asic)].SB_Port();
      uint16_t FA_ID=ASICid_from_ID(asic);

      if(!m_FPGAs[FPGAid_from_ID(asic)].IsScEnabled()){
      //    printf(" [skipped]\n");
          return FE_SUCCESS;
      }
      if(SB_ID!=m_SB_number){
      //    printf(" [skipped]\n");
          return FE_SUCCESS;
      }
      //printf("\n");

      cm_msg(MINFO, "setup_mutrig" , "Configuring MuTRiG asic %s/Settings/ASICs/%i/: Mapped to SB%u.%u  ASIC #%d", m_odb_prefix, asic,SB_ID,SP_ID,FA_ID);


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

int MutrigFEB::ReadBackRunState(uint16_t FPGA_ID){
   uint32_t val[2];
   int status=m_mu.FEBsc_read(FPGA_ID, val, 2, 0xfffe);
   if(status!=SUCCESS) return status;
   //val[0] is reset_bypass register
   //val[1] is reset_bypass payload
   std::string cmd_name="undefined";
   reset cmds;
   for(auto it : cmds.commands){
      if(it.second.command==val[0] & 0xff) cmd_name=it.first;
   }
   std::string state_str="undefined";
   switch(val[0]&0xffff0000){
      case 1<<0:
      state_str="RUN_STATE_BITPOS_IDLE"; 
      break;
      case 1<<1:
      state_str="RUN_STATE_BITPOS_PREP"; 
      break;
      case 1<<2:
      state_str="RUN_STATE_BITPOS_SYNC"; 
      break;
      case 1<<3:
      state_str="RUN_STATE_BITPOS_RUNNING"; 
      break;
      case 1<<4:
      state_str="RUN_STATE_BITPOS_TERMINATING"; 
      break;
      case 1<<5:
      state_str="RUN_STATE_BITPOS_LINK_TEST"; 
      break;
      case 1<<6:
      state_str="RUN_STATE_BITPOS_SYNC_TEST"; 
      break;
      case 1<<7:
      state_str="RUN_STATE_BITPOS_RESET"; 
      break;
      case 1<<8:
      state_str="RUN_STATE_BITPOS_OUT_OF_DAQ"; 
      break;
      default:
      state_str="-broken-";
   }
   char path[255];
   sprintf(path, "%s/Variables/FEB Run State/Bypass command", m_odb_prefix);
   if((status = db_set_value_index(m_hDB, 0, path, cmd_name.c_str(), cmd_name.length(),FPGA_ID, TID_STRING,false))!=DB_SUCCESS) return status;

   sprintf(path, "%s/Variables/FEB Run State/Run state", m_odb_prefix);
   if((status = db_set_value_index(m_hDB, 0, path, state_str.c_str(), state_str.length(),FPGA_ID, TID_STRING,false))!=DB_SUCCESS) return status;



}

int MutrigFEB::ReadBackCounters(uint16_t FPGA_ID){
   auto rpc_ret=m_mu.FEBsc_NiosRPC(FPGA_ID,0x0105,{});
   //retrieve results
   uint32_t* val=new uint32_t[rpc_ret*4*3]; //nASICs * 4 counterbanks * 3 words
   INT val_size = sizeof(DWORD);
   printf("RPC return: %u\n",rpc_ret);
   m_mu.FEBsc_read(FPGA_ID, val, rpc_ret*4*3 , (uint32_t) m_mu.FEBsc_RPC_DATAOFFSET);
   printf("done reading:\n");
   for(int i=0;i<rpc_ret*4*3;i++){
      printf("%2.2d %2.2d: %8.8x\n",i/(3*4),(i/3)%4,val[i]);
   }
   //store in midas
   INT status;
   int index=0;
   printf("done reading: odb:%s\n",m_odb_prefix);
   char path[255];
   printf("odb var:%s\n",path);
   for(int nASIC=0;nASIC<rpc_ret;nASIC++){
       printf("writing %d\n",nASIC);
       sprintf(path,"%s/Variables/Counters/nHits",m_odb_prefix);
       if((status=db_set_value_index(m_hDB, 0, path, &val[index], val_size, nASIC, TID_DWORD, FALSE))!=DB_SUCCESS) return status;
       index+=1;
       sprintf(path,"%s/Variables/Counters/Time",m_odb_prefix);
       if((status=db_set_value_index(m_hDB, 0, path, &val[index], val_size, nASIC, TID_DWORD, FALSE))!=DB_SUCCESS) return status;
       index+=2;
       sprintf(path,"%s/Variables/Counters/nBadFrames",m_odb_prefix);
       if((status=db_set_value_index(m_hDB, 0, path, &val[index], val_size, nASIC, TID_DWORD, FALSE))!=DB_SUCCESS) return status;
       index+=1;
       sprintf(path,"%s/Variables/Counters/nFrames",m_odb_prefix);
       if((status=db_set_value_index(m_hDB, 0, path, &val[index], val_size, nASIC, TID_DWORD, FALSE))!=DB_SUCCESS) return status;
       index+=2;
       sprintf(path,"%s/Variables/Counters/nErrorsLVDS",m_odb_prefix);
       if((status=db_set_value_index(m_hDB, 0, path, &val[index], val_size, nASIC, TID_DWORD, FALSE))!=DB_SUCCESS) return status;
       index+=1;
       sprintf(path,"%s/Variables/Counters/nWordsLVDS",m_odb_prefix);
       if((status=db_set_value_index(m_hDB, 0, path, &val[index], val_size, nASIC, TID_DWORD, FALSE))!=DB_SUCCESS) return status;
       index+=2;
       sprintf(path,"%s/Variables/Counters/nErrorsPRBS",m_odb_prefix);
       if((status=db_set_value_index(m_hDB, 0, path, &val[index], val_size, nASIC, TID_DWORD, FALSE))!=DB_SUCCESS) return status;
       index+=1;
       sprintf(path,"%s/Variables/Counters/nWordsPRBS",m_odb_prefix);
       if((status=db_set_value_index(m_hDB, 0, path, &val[index], val_size, nASIC, TID_DWORD, FALSE))!=DB_SUCCESS) return status;
       index+=2;
   }

   delete[] val;
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
		printf("LVDS_waitforall_sticky -> %u.%u\n",FEB.SB_Number(),FEB.SB_Port());
		_this->setWaitForAllSticky(FEB.SB_Port(),bval);
	}
   }

   if (std::string(key.name) == "mask") {
      printf("mask: GetNumASICs()=%d\n",_this->GetNumASICs());
      BOOL* barray=new BOOL[_this->GetNumASICs()];
      INT  barraysize=sizeof(BOOL)*_this->GetNumASICs();
      db_get_data(hDB,hKey,barray,&barraysize,TID_BOOL);
      
      for(int i=0;i<_this->GetNumASICs();i++){
         if(_this->m_FPGAs[_this->FPGAid_from_ID(i)].IsScEnabled()==false) continue;
         if(_this->m_FPGAs[_this->FPGAid_from_ID(i)].SB_Number()!=_this->m_SB_number) continue;
         //printf("mask(%d) -> %u -> %u.%u\n",i,_this->FPGAid_from_ID(i),_this->m_FPGAs[_this->FPGAid_from_ID(i)].SB_Port());
         _this->setMask(_this->m_FPGAs[_this->FPGAid_from_ID(i)].SB_Port(),barray[i]);
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
            cm_msg(MINFO, "MutrigFEB::on_settings_changed", "reset_datapath");
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
        cm_msg(MINFO, "MutrigFEB::on_settings_changed", "reset_asics");
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
        cm_msg(MINFO, "MutrigFEB::on_settings_changed", "reset_lvds");
        for(auto FEB: _this->m_FPGAs){
        if(!FEB.IsScEnabled()) continue; //skip disabled
        if(FEB.SB_Number()!=_this->m_SB_number) continue; //skip commands not for me
        cm_msg(MINFO, "MutrigFEB::on_settings_changed", "reset_asics");
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
        cm_msg(MINFO, "MutrigFEB::on_settings_changed", "Updating reset skew settings");
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
            BOOL vals[_this->nModulesPerFEB()];
            vals[0]=cphase[i]; vals[1]=cphase[i+1];
            _this->setResetSkewCphase(_this->m_FPGAs[i].SB_Port(),vals);
            vals[0]=cdelay[i]; vals[1]=cdelay[i+1];
            _this->setResetSkewCdelay(_this->m_FPGAs[i].SB_Port(),vals);
            INT ivals[_this->nModulesPerFEB()];
            ivals[0]=phases[i]; ivals[1]=phases[i+1];
            _this->setResetSkewPhases(_this->m_FPGAs[i].SB_Port(),ivals);
        }
    }
}

//Helper functions
uint32_t reg_setBit  (uint32_t reg_in, uint8_t bit, bool value=true){
    if(value)
        return (reg_in | 1<<bit);
    else
        return (reg_in & (~(1<<bit)));
}
uint32_t reg_unsetBit(uint32_t reg_in, uint8_t bit){return reg_setBit(reg_in,bit,false);}

bool reg_getBit(uint32_t reg_in, uint8_t bit){
    return (reg_in & (1<<bit)) != 0;
}

uint32_t reg_getRange(uint32_t reg_in, uint8_t length, uint8_t offset){
    return (reg_in>>offset) & ((1<<length)-1);
}
uint32_t reg_setRange(uint32_t reg_in, uint8_t length, uint8_t offset, uint32_t value){
    return (reg_in & ~(((1<<length)-1)<<offset)) | ((value & ((1<<length)-1))<<offset);
}


//MutrigFEB registers and functions

/**
* Use emulated mutric on fpga for config
*/
void MutrigFEB::setDummyConfig(uint16_t FPGA_ID, bool dummy){
    printf("MutrigFEB::setDummyConfig(%d)=%d\n",FPGA_ID,dummy);
    uint32_t val;

    //TODO: shadowing should know about broadcast FPGA ID
    //TODO: implement pull from FPGA when shadow value is not stored
    //m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
    //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG readback=%8.8x\n",FPGA_ID,val);
    val=m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG];

    val=reg_setBit(val,0,dummy);
    printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG new=%8.8x\n",FPGA_ID,val);
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
    printf("MutrigFEB::setDummyData_Enable(%d)=%d\n",FPGA_ID,dummy);
    uint32_t val;

    val=m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG];
    //m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
    //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG readback=%8.8x\n",FPGA_ID,val);

    val=reg_setBit(val,1,dummy);
    printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG new=%8.8x\n",FPGA_ID,val);
    m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG, m_ask_sc_reply);
    m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG]=val;
}

void MutrigFEB::setDummyData_Fast(uint16_t FPGA_ID, bool fast)
{
    printf("MutrigFEB::setDummyData_Fast(%d)=%d\n",FPGA_ID,fast);
    uint32_t  val;
    val=m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG];
    //m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
    //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG readback=%8.8x\n",FPGA_ID,val);

    val=reg_setBit(val,2,fast);
    printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG new=%8.8x\n",FPGA_ID,val);
    m_mu.FEBsc_write(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG, m_ask_sc_reply);
    m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG]=val;
}

void MutrigFEB::setDummyData_Count(uint16_t FPGA_ID, int n)
{
    if(n > 255) n = 255;
    printf("MutrigFEB::setDummyData_Count(%d)=%d\n",FPGA_ID,n);
    uint32_t  val;
    val=m_reg_shadow[FPGA_ID][FE_DUMMYCTRL_REG];
    //m_mu.FEBsc_read(FPGA_ID, &val, 1 , (uint32_t) FE_DUMMYCTRL_REG);
    //printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG readback=%8.8x\n",FPGA_ID,val);
    val=reg_setRange(val, 9, 3, n);

    printf("MutrigFEB(%d)::FE_DUMMYCTRL_REG new=%8.8x\n",FPGA_ID,val);
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
    printf("MutrigFEB::setPRBSDecoderDisable(%d)=%d\n",FPGA_ID,disable);
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
    printf("MutrigFEB::setWaitForAll(%d)=%d\n",FPGA_ID,value);
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
    printf("MutrigFEB::setWaitForAllSticky(%d)=%d\n",FPGA_ID,value);
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
