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

#include <iostream>
#include <fstream>

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
    odb FEBsSettings(odb_prefix + "/Settings/FEBS");
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

void MupixFEB::SetTDACs() {

    for (int asic = 0; asic < GetNumASICs(); asic++) {
        odb TDACsSettings(odb_prefix + "/Settings/TDACs/" + std::to_string(asic));
        std::string TDACFILE = TDACsSettings["TDACFILE"];
        std::ifstream data(TDACFILE);
        std::string line;
        std::map<std::string, std::vector<uint32_t>> parsedCsv;
        bool firstLine = true;
        while ( std::getline(data, line) )
        {
            if (firstLine) {
                firstLine = false;
            } else {
                std::stringstream lineStream(line);
                std::string cell;
                std::vector<uint32_t> parsedRow;
                std::string firstValue = "-999";
                while(std::getline(lineStream, cell, ','))
                {
                    if ( firstValue == "-999" ) {
                        firstValue = cell;
                    } else {
                        parsedRow.push_back(std::stoi(cell));
                    }
                }
                parsedCsv.insert(std::pair<std::string, std::vector<uint32_t>>(firstValue, parsedRow));
            }
        }
        TDACsJSON.push_back(parsedCsv);
    }   
}

//ASIC configuration:
//Configure all asics under prefix (e.g. prefix="/Equipment/Mupix")
int MupixFEB::ConfigureASICs(){
    
    printf("MupixFEB::ConfigureASICs()\n");
    cm_msg(MINFO, "MupixFEB" , "Configuring sensors under prefix %s/Settings/ASICs/", odb_prefix.c_str());

    // write lvds mask from ODB to each feb
    for (auto feb : febs){
        odb FEBsSettings(odb_prefix + "/Equipment/Mupix/Settings/FEBS/" + std::to_string(feb.GetLinkID()));
        feb_sc.FEB_write(feb, MP_LVDS_LINK_MASK_REGISTER_W, (uint32_t) FEBsSettings["MP_LVDS_LINK_MASK"]);
        feb_sc.FEB_write(feb, MP_LVDS_LINK_MASK2_REGISTER_W, (uint32_t) FEBsSettings["MP_LVDS_LINK_MASK2"]);
    }
    
    // configure each asic
    int status = mupix::midasODB::MapForEachASIC(odb_prefix, [this](mupix::MupixConfig* config, uint32_t asic){
//                 if ( asic != 3 ) return 0;
        uint32_t rpc_status;
        //bool TDACsNotFound = false;
        //char set_str[255];

        // get settings from ODB for TDACs 
        // TODO: Has to move!!!
        odb swbSettings("/Equipment/Switching/Settings");
        bool useTDACs = swbSettings["MupixSetTDACConfig"];
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
                odb_prefix.c_str(),asic,FPGAid_from_ID(asic),SB_ID,SP_ID,FA_ID);

        // TODO: There is a lot of copy/paste in the following - I guess we can condense this
        // down a lot with a well chosen function call
        uint32_t bitpattern_m;
        vector<vector<uint32_t> > payload_m;
        vector<uint32_t> payload;
        uint32_t spi_busy;
        uint32_t count = 0;
        uint32_t limit = 5;
        try {

            payload_m.push_back(vector<uint32_t>(reinterpret_cast<uint32_t*>(config->bitpattern_w),reinterpret_cast<uint32_t*>(config->bitpattern_w)+config->length_32bits));

            for(uint32_t j = 0; j<payload_m.at(0).size();j++){
                bitpattern_m=0;
                for(short i=0; i<32; i++){
                    bitpattern_m|= ((payload_m.at(0).at(j)>>i) & 0b1)<<(31-i);
                }
                payload.push_back(bitpattern_m);
            }

            for(uint32_t j = 0; j<payload.size();j++){
                std::cout<<std::hex<<payload.at(j)<<std::endl;
            }

            // ToDo: Col Test Tdac bits from file
            for(int i=0; i<85; i++){
                payload.push_back(0x00000000);
            }

            // mask all chips but not this one
            // TODO: make this correct
            uint32_t chip_select_mask = 0xfff; //all chips masked (12 times 1)
            uint16_t pos = ASICid_from_ID(asic);
            bool isTelescope = false; // TODO: make this somehow dynamic for the telescope setup
	        if ( asic == 3 && isTelescope ) pos = 3;
            chip_select_mask &= ((~0x1u) << pos);
            printf("chip_select_mask %04x\n", chip_select_mask);
            for (int i = 0; i < pos; ++i)
                chip_select_mask |= (0x1 << i);
            printf("chip_select_mask %04x\n", chip_select_mask);

            // check if FEB is busy
            rpc_status=FEB_REPLY_SUCCESS;
            feb_sc.FEB_read(FEB, MP_CTRL_SPI_BUSY_REGISTER_R, spi_busy);
            while(spi_busy==1 && count < limit){
                sleep(1);
                feb_sc.FEB_read(FEB,MP_CTRL_SPI_BUSY_REGISTER_R,spi_busy);
                count++;
                cm_msg(MINFO, "MupixFEB", "Mupix config spi busy .. waiting");
            }

            if (count == limit) {
                std::cout<<"Timeout"<<std::endl;
                cm_msg(MERROR, "setup_mupix", "FEB Mupix SPI timeout");
            } else { // do the SPI writing 
                // TODO: make this correct
                feb_sc.FEB_write(FEB, MP_CTRL_CHIP_MASK_REGISTER_W, 0x0);//chip_select_mask); //
                // TODO: include headers for addr.
                feb_sc.FEB_write(FEB, MP_CTRL_SLOW_DOWN_REGISTER_W, 0x0000000F); // SPI slow down reg
                feb_sc.FEB_write(FEB, MP_CTRL_ENABLE_REGISTER_W, 0x00000FC0); // reset Mupix config fifos
                feb_sc.FEB_write(FEB, MP_CTRL_ENABLE_REGISTER_W, 0x00000000);
                feb_sc.FEB_write(FEB, MP_CTRL_INVERT_REGISTER_W, 0x00000003); // idk, have to look it up
                // We now only write the default configuration for testing
                rpc_status = feb_sc.FEB_write(FEB, MP_CTRL_ALL_REGISTER_W, payload,true);
            }
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
      
        // check if we also want to write the TDACs
        if (useTDACs) {
            std::cout << "Write TDACs" << "\n";
            uint32_t curNBits = 0;
            uint32_t curWord = 0;
            // loop over keys of the tdacs dict for the current asic
            // {"0": ["0x0", "0x0", "0x0", "0x0", ...],
            // {"1": ["0x0", "0x0", "0x0", "0x0", ...],
            // {"5": ["0x0", "0x0", "0x0", "0x0", ...],
            // from the key we get the col value for the masking by 6*key+6
            // the row values for the masking (512 bits) are stored in 32b words in the json file
            
            for (auto it = GetTDACsJSON().at(asic).begin(); it != GetTDACsJSON().at(asic).end(); it++) {
                std::cout << "KEY: " << it->first << "\n";
                // check if FEB is busy
                feb_sc.FEB_read(FEB, MP_CTRL_SPI_BUSY_REGISTER_R, spi_busy);
                count = 0;
                while ( spi_busy==1 && count < limit ) {
                    sleep(1);
                    feb_sc.FEB_read(FEB, MP_CTRL_SPI_BUSY_REGISTER_R, spi_busy);
                    count++;
                    cm_msg(MINFO, "MupixFEB", "Mupix config spi busy .. waiting");
                }
                if (count == limit) {
                    std::cout << "Timeout" << std::endl;
                    cm_msg(MERROR, "setup_mupix", "FEB Mupix SPI timeout for TDAC writing");
                } else {
                    // first we write the row values from the value
                    for ( uint32_t v : it->second ) {
                        std::cout << "VALUE: " << v << "\n";
                        feb_sc.FEB_write(FEB, MP_CTRL_TDAC_REGISTER_W, v);
                    }
                    feb_sc.FEB_write(FEB, MP_CTRL_ENABLE_REGISTER_W, reg_setBit(0x0,WR_TDAC_BIT,true));
                    feb_sc.FEB_write(FEB,MP_CTRL_ENABLE_REGISTER_W,0x0);
                    
                    // now we write the 128*7b col values where we write on col (key) at the time
                    curWord = 0;
                    curNBits = 0;
                    for ( int i = 0; i <= 127; i++ ) {
                        for ( int b = 0; b < 7; b++ ) {
                            curNBits++;
                            if (b == 6 && it->first == std::to_string(i)) {
                                curWord = curWord | (1 << curNBits);
                            }
                            if (curNBits == 32) {
                                feb_sc.FEB_write(FEB, MP_CTRL_COL_REGISTER_W, curWord);
                                curWord = 0;
                                curNBits = 0;
                            }
                        }
                    }
                    feb_sc.FEB_write(FEB, MP_CTRL_ENABLE_REGISTER_W, reg_setBit(0x0,WR_COL_BIT,true));
                    feb_sc.FEB_write(FEB,MP_CTRL_ENABLE_REGISTER_W,0x0);
                }
            }
        }

        // reset lvds links
        feb_sc.FEB_write(FEB, MP_RESET_LVDS_N_REGISTER_W, 0x0);
        feb_sc.FEB_write(FEB, MP_RESET_LVDS_N_REGISTER_W, 0x1);

        sleep(2);
        
        return FE_SUCCESS;//note: return of lambda function
    });//MapForEach

    return status; //status of foreach function, SUCCESS when no error.
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
    //cm_msg(MINFO, "MupixFEB::ReadBackLVDSNumHits" , "Implement Me");
    return 0;
}

uint32_t MupixFEB::ReadBackLVDSNumHitsInMupixFormat(mappedFEB & FEB, uint16_t LVDS_ID)
{
    //cm_msg(MINFO, "MupixFEB::ReadBackLVDSNumHitsInMupixFormat" , "Implement Me");
    return 0;
}

DWORD* MupixFEB::ReadLVDSCounters(DWORD* pdata, mappedFEB & FEB)
{
    for(uint32_t i=0; i<64; i++){ 

        // TODO: intrun fix for lvds configuration
        if (i>=lvds_links_per_feb) continue;
        // Link ID
        *pdata++ = i;
        // read lvds status
        *pdata++ = ReadBackLVDSStatus(FEB, i);
        // number of hits from link
        *pdata++ = ReadBackLVDSNumHits(FEB, i);
        // number of hits from link in mupix format
        *pdata++ = ReadBackLVDSNumHitsInMupixFormat(FEB, i);

    };
    return pdata;
}
