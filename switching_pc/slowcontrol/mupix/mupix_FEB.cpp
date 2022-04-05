/********************************************************************\

  Name:         Mupix_FEB.h
  Created by:   Konrad Briggl

Contents:       Definition of functions to talk to a mupix-based FEB. Designed to no be derived into a SciFi_FEB and a Tiles_FEB class where subdetector specific actions are defined.
		Here: Definition of basic things for mupix-configuration & datapath settings

\********************************************************************/

#include "mupix_FEB.h"
#include "midas.h"
#include "odbxx.h"
#include "mfe.h" //for set_equipment_status
#include "odbxx.h"

#include "../include/feb.h"
using namespace mu3e::daq;

#include "mudaq_device.h"
#include "mupix_config.h"
#include "mupix_midasodb.h"
#include <thread>
#include <chrono>
#include <vector>

#include <iostream>
#include <fstream>
#include <istream>
using midas::odb;

#include "default_config_mupix.h" //TODO avoid this, reproduce configure routine from chip dacs

//offset for registers on nios SC memory
#define SC_REG_OFFSET 0xff60
#define FE_DUMMYCTRL_REG       (SC_REG_OFFSET+0x8)
#define FE_DPCTRL_REG          (SC_REG_OFFSET+0x9)
#define FE_SUBDET_RESET_REG    (SC_REG_OFFSET+0xa)
#define FE_SPIDATA_ADDR		0

//status flags from FEB
#define FEB_REPLY_SUCCESS 0
#define FEB_REPLY_ERROR   1

//Mapping to physical ports of switching board.
uint16_t MupixFEB::ASICsPerFEB() const {
    odb FEBsSettings(pixel_odb_prefix + "/Settings/FEBS");
    return (uint16_t) FEBsSettings["ASICsPerFEB"];
}

uint16_t MupixFEB::FPGAid_from_ID(int asic) const {
    return asic/ASICsPerFEB();
}

uint16_t MupixFEB::ASICid_from_ID(int asic) const {
    return asic%ASICsPerFEB();
}

uint16_t MupixFEB::GetNumASICs() const {
    return febs.size()*ASICsPerFEB();
}

//ASIC configuration:
//Configure all asics under prefix (e.g. prefix="/Equipment/Mupix")
int MupixFEB::ConfigureASICs(){

    printf("MupixFEB::ConfigureASICs()\n");
    cm_msg(MINFO, "MupixFEB" , "Configuring sensors under prefix %s/Settings/ASICs/", pixel_odb_prefix.c_str());

    // write lvds mask from ODB to each feb
    for (auto feb : febs){
        odb FEBsSettings(pixel_odb_prefix + "/Settings/FEBS/" + std::to_string(feb.GetLinkID()));
        feb_sc.FEB_write(feb, MP_LVDS_LINK_MASK_REGISTER_W, (uint32_t) FEBsSettings["MP_LVDS_LINK_MASK"]);
        feb_sc.FEB_write(feb, MP_LVDS_LINK_MASK2_REGISTER_W, (uint32_t) FEBsSettings["MP_LVDS_LINK_MASK2"]);
        feb_sc.FEB_write(feb, MP_CTRL_SPI_ENABLE_REGISTER_W, 0x00000001);
        feb_sc.FEB_write(feb, MP_CTRL_SLOW_DOWN_REGISTER_W, 0x0000000F);
    }

    // configure each asic
    int status = mupix::midasODB::MapForEachASIC(pixel_odb_prefix, [this](mupix::MupixConfig* config, uint32_t asic){
//                 if ( asic != 3 ) return 0;
        uint32_t rpc_status;
        //bool TDACsNotFound = false;
        //char set_str[255];

        // get settings from ODB for TDACs 
        // TODO: Has to move!!!
        odb swbSettings(odb_prefix + "/Settings");
        uint32_t MupixChipToConfigure = swbSettings["MupixChipToConfigure"];
        if ( MupixChipToConfigure != 999 && asic != MupixChipToConfigure ) {
            printf(" [skipped]\n");
            return FE_SUCCESS;
        }
        
        //mapping
        auto FEB = febs[FPGAid_from_ID(asic)];
        uint16_t SB_ID=FEB.SB_Number();
        uint16_t SP_ID=FEB.SB_Port();
        uint16_t FA_ID=ASICid_from_ID(asic);

        if(!FEB.IsScEnabled()){
            printf(" [skipped]\n");
            return FE_SUCCESS;
        }
        if(SB_ID!= SB_number){
            printf(" [skipped]\n");
            return FE_SUCCESS;
        } //TODO

        cm_msg(MINFO, "MupixFEB",
                "Configuring sensor %s/Settings/ASICs/%i/: Mapped to FEB%u -> SB%u.%u  ASIC #%d",
                pixel_odb_prefix.c_str(),asic,FPGAid_from_ID(asic),SB_ID,SP_ID,FA_ID);

        // TODO: There is a lot of copy/paste in the following - I guess we can condense this
        // down a lot with a well chosen function call
        uint32_t bitpattern_m;
        vector<vector<uint32_t> > payload_m;
        vector<uint32_t> payload;

        try {

            payload_m.push_back(vector<uint32_t>(reinterpret_cast<uint32_t*>(config->bitpattern_w),reinterpret_cast<uint32_t*>(config->bitpattern_w)+config->length_32bits));

            for(uint32_t j = 0; j<payload_m.at(0).size();j++){
                bitpattern_m=0;
                for(short i=0; i<32; i++){
                    bitpattern_m|= ((payload_m.at(0).at(j)>>i) & 0b1)<<(31-i);
                }
                payload.push_back(bitpattern_m);
            }
            int size = payload.size();
            payload[size-1] = 0x2900303;//TOFIX: why different?

            std::cout << "Payload:\n";
            for(uint32_t j = 0; j<payload.size();j++){
                std::cout<<std::hex<<payload.at(j)<<std::endl;
            }

            //uint32_t chip_select_mask = 0xfff; //all chips masked (12 times 1)
            uint16_t pos = ASICid_from_ID(asic);
            bool isTelescope = false; // TODO: make this somehow dynamic for the telescope setup
            rpc_status = feb_sc.FEB_write(FEB, MP_CTRL_COMBINED_START_REGISTER_W + asic, payload,true);

        } catch(std::exception& e) {
            cm_msg(MERROR, "setup_mupix", "Communication error while configuring MuPix %d: %s", asic, e.what());
            set_equipment_status(equipment_name.c_str(), "SB-FEB Communication error", "red");
            return FE_ERR_HW; //note: return of lambda function
        }

        if(rpc_status!=FEB_REPLY_SUCCESS){
            //configuration mismatch, report and break foreach-loop
            set_equipment_status(equipment_name.c_str(),  "MuPix config failed", "red");
            cm_msg(MERROR, "setup_mupix", "MuPix configuration error for ASIC %i", asic);
            return FE_ERR_HW;//note: return of lambda function
        }

        // reset lvds links
        feb_sc.FEB_write(FEB, MP_RESET_LVDS_N_REGISTER_W, 0x0);
        feb_sc.FEB_write(FEB, MP_RESET_LVDS_N_REGISTER_W, 0x1);

        sleep(2);
        
        return FE_SUCCESS;//note: return of lambda function
    });//MapForEach

    return status; //status of foreach function, SUCCESS when no error.
}

void read_tdac_file(vector<uint32_t>& vec, std::string path) { 
  std::ifstream file;
  file.open(path);
  file.read(reinterpret_cast<char*>(&vec[0]), 256*64*sizeof(uint32_t));
  file.close();
}

    // TODO after cosmic run: check how fast this is, improve it
int MupixFEB::ConfigureTDACs(){
    int status = feb_sc.ERRCODES::OK;

            // Tdac writing for a chip would work like this:
            // write 32 bit words, 4* 8bit tdac in each word
            // order: 
            /*
                col 0 -> 255, starting with col 0, physical col addr. 
                row 0 -> row 255 for each col, starting with row0, physical row addr.

                example:
                start with col 0 ..
                (32 bit)  : [8 bit tdac, 8 bit tdac, 8 bit tdac, 8 bit tdac]
                word 0    : [row 3     , row 2     , row 1     , row 0     ]
                word 1    : [row 7     , row 6     , row 5     , row 4     ]
                word 2    : [row 11    , row 10    , row 9     , row 8     ]
                ...
                word 63   : [row 255   , row 254   , row253    , row 252   ]
                now col 1 ..
                word 64   : [row 3     , row 2     , row 1     , row 0     ]
                ...
            */
    printf("MupixFEB::ConfigureTDACs()\n");
    cm_msg(MINFO, "MupixFEB" , "Configuring TDACS");
    std::vector<uint32_t> test;
    std::vector<std::vector<uint8_t>> pages_remaining; // vector of FEBs containing vector of sensors, containing number of remaining pages until that sensor is configured

    // what is the best way to do this ? .. this looks ugly
    std::vector<std::vector<std::vector<uint32_t>>> tdac_pages; // vector of FEBs containing a vector of sensors, containing a vector of tdac values of that page
    std::vector<uint8_t> pages_remaining_this_chip;
    uint8_t N_PAGES_PER_CHIP;
    uint32_t PAGESIZE = 128;  // todo get these things from firmware constants
    uint8_t current_page = 0;
    uint32_t N_free_pages = 0;
    bool allDone = false;
    uint16_t N_CHIPS = 0;
    uint16_t internal_febID = 0;
    std::string path;

    // preparation loop
    for (auto feb : febs){
        // set all febs to use spi with a slow down of F
        feb_sc.FEB_write(feb, MP_CTRL_SPI_ENABLE_REGISTER_W, 0x00000001);
        feb_sc.FEB_write(feb, MP_CTRL_SLOW_DOWN_REGISTER_W, 0x0000000F);
        // run configuration init    (TODO: seperate config init from writing 0 to all tdacs in firmware)
        feb_sc.FEB_write(feb, MP_CTRL_RESET_REGISTER_W, 0x00000001);

        N_CHIPS = ASICsPerFEB();
        N_PAGES_PER_CHIP=128; // will this be the same for all febs ( -> get from firmware constants) or not (-> read register from FEB) ?
        std::vector<std::vector<uint32_t>> tdac_page_this_feb;
        for(uint32_t i = 0; i < N_CHIPS; i++){
            pages_remaining_this_chip.push_back(N_PAGES_PER_CHIP);
            std::vector<uint32_t> tdac_chip(64*256);
            odb FEBsSettings(pixel_odb_prefix + "/Settings/TDACS/" + std::to_string(i+(internal_febID+N_CHIPS)));
            path = FEBsSettings["TDACFILE"];
            read_tdac_file(tdac_chip, "path");
            tdac_page_this_feb.push_back(tdac_chip);
        }
        tdac_pages.push_back(tdac_page_this_feb);
        pages_remaining.push_back(pages_remaining_this_chip);
        pages_remaining_this_chip.clear();
        internal_febID++;
    }

    while (! allDone){
        internal_febID = 0;
        allDone = false;

        for (auto feb : febs){

            // get number of free tdac pages for this feb
            feb_sc.FEB_read(feb,MP_CTRL_N_FREE_PAGES_REGISTER_R, N_free_pages);

            // while the feb has space left ..
            while(!allDone && N_free_pages > 0){
                // Write one page for every chip
                allDone = true;
                for (uint32_t chip = 0; chip<pages_remaining.at(internal_febID).size(); chip++){
                    if (pages_remaining.at(internal_febID).at(chip) != 0){
                        allDone = false;
                        if(N_free_pages > 0){
                            current_page = N_PAGES_PER_CHIP-pages_remaining.at(internal_febID)[chip];
                            pages_remaining.at(internal_febID)[chip] = pages_remaining.at(internal_febID)[chip] - 1;
                            std::vector<uint32_t> tdac_page(PAGESIZE);
                            // how to do this wihout copy ? 
                            tdac_page = std::vector<uint32_t>(tdac_pages.at(internal_febID).at(chip).begin() + current_page*PAGESIZE, tdac_pages.at(internal_febID).at(chip).begin() + (current_page+1)*PAGESIZE);
                            feb_sc.FEB_write(feb, MP_CTRL_TDAC_START_REGISTER_W + chip, tdac_page, true, false);
                            N_free_pages--;
                        } else {break;}
                    }
                }
                if(allDone) break;
            }
            internal_febID++;
        }
    }
    return status;
}

//MIDAS callback function for FEB register Setter functions
void MupixFEB::on_settings_changed(odb o, void * userdata)
{
    std::string name = o.get_name();

    cm_msg(MINFO, "MupixFEB::on_settings_changed", "Setting changed (%s)", name.c_str());

    //MupixFEB* _this=static_cast<MupixFEB*>(userdata);
    
    BOOL bval;

    if (name == "dummy_config") {
        bval = o;
        
        cm_msg(MINFO, "MupixFEB::on_settings_changed", "Set dummy_config %d", bval);
        // TODO: do something here
        //     _this->setDummyConfig(MupixFEB::FPGA_broadcast_ID,value);
   }

   if (name == "reset_asics") {
      bval = o;
      if(bval){
         cm_msg(MINFO, "MupixFEB::on_settings_changed", "reset_asics");
         // TODO: do something here
//         _this->chipReset(MupixFEB::FPGA_broadcast_ID);
         o = FALSE; // reset flag in ODB
      }
   }
   
   if (name == "reset_boards") {
      bval = o;
      if(bval){
         cm_msg(MINFO, "MupixFEB::on_settings_changed", "reset_boards");
         // TODO: do something here
//         _this->chipReset(MupixFEB::FPGA_broadcast_ID);
        o = FALSE; // reset flag in ODB
      }
   }

}

unsigned char reverse(unsigned char b) {
   b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
   b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
   b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
   return b;
}

uint32_t MupixFEB::ReadBackLVDSStatus(mappedFEB & FEB, uint16_t LVDS_ID)
{
   //skip disabled fibers
    if(!FEB.IsScEnabled())
        return 0;

    //skip commands not for this SB
    if(FEB.SB_Number()!=SB_number)
        return 0;

    if(!FEB.GetLinkStatus().LinkIsOK())
        return 0;    
    
    uint32_t val;
    feb_sc.FEB_read(FEB, MP_LVDS_STATUS_START_REGISTER_W + LVDS_ID, val);
    
    return val;
}

uint32_t MupixFEB::ReadBackLVDSNumHits(mappedFEB & FEB, uint16_t LVDS_ID)
{
    //TODO Read a hit counter here
    //cm_msg(MINFO, "MupixFEB::ReadBackLVDSNumHits" , "Implement Me");
    return 0;
}


DWORD* MupixFEB::ReadLVDSCounters(DWORD* pdata, mappedFEB & FEB)
{
    for(uint32_t i=0; i<64; i++){ 

        // TODO: intrun fix for lvds configuration: The FEB should know how many links it has...
        if (i>= MAX_LVDS_LINKS_PER_FEB) continue;
        // Link ID
        *pdata++ = i;
        // read lvds status
        *pdata++ = ReadBackLVDSStatus(FEB, i);
        // number of hits from link
        *pdata++ = ReadBackLVDSNumHits(FEB, i);


    };
    return pdata;
}


DWORD* MupixFEB::ReadLVDSforPSLS(DWORD* pdata, mappedFEB & FEB)
{
    std::vector<uint32_t> status(MAX_LVDS_LINKS_PER_FEB);
    feb_sc.FEB_read(FEB, MP_LVDS_STATUS_START_REGISTER_W, status);

    std::vector<uint32_t> histos(MAX_LVDS_LINKS_PER_FEB*4);
    // TODO: Uncomment when register is defined
    //feb_sc.FEB_read(FEB, MP_HIT_ARRIVAL_START_REGISTER, histos);


    for(uint32_t i=0; i<64; i++){ 
        if (i>=MAX_LVDS_LINKS_PER_FEB) continue;
        // Link ID
        *pdata++ = i;  
        *pdata++ = status[i];
        *pdata++ = ReadBackLVDSNumHits(FEB, i);
        *pdata++ = histos[i*4];
        *pdata++ = histos[i*4+1];
        *pdata++ = histos[i*4+2]; 
        *pdata++ = histos[i*4+3];
    }
    return pdata;
}
