/* Constants and definitions for MIDAS banks */


#ifndef MU3EBANKS_H
#define MU3EBANKS_H

#include <array>
#include <string>
#include <string>

#include "odbxx.h"

#include "link_constants.h"

using std::array;
using std::string;
using std::to_string;

using midas::odb;

////////////// Switching board

//// SSFE
constexpr int per_fe_SSFE_size = 26;
const array<const string, MAX_N_SWITCHINGBOARDS> ssfe = {"SCFE","SUFE","SDFE","SFFE"};
const array<const string, MAX_N_SWITCHINGBOARDS> ssfenames = {"Names SCFE","Names SUFE","Names SDFE","Names SFFE"};

void create_ssfe_names_in_odb(odb & settings, int switch_id){
    string namestr = ssfenames[switch_id];

    int bankindex = 0;

    for(uint32_t i=0; i < N_FEBS[switch_id]; i++){
        string feb = "FEB" + to_string(i);
        string * s = new string(feb);
        (*s) += " Index";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Arria Temperature";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " MAX Temperature";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " SI1 Temperature";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " SI2 Temperature";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " ext Arria Temperature";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " DCDC Temperature";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Voltage 1.1";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Voltage 1.8";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Voltage 2.5";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Voltage 3.3";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Voltage 20";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly1 Temperature";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly1 Voltage";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly1 RX1 Power";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly1 RX2 Power";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly1 RX3 Power";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly1 RX4 Power";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly1 Alarms";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly2 Temperature";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly2 Voltage";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly2 RX1 Power";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly2 RX2 Power";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly2 RX3 Power";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly2 RX4 Power";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly2 Alarms";
        settings[namestr][bankindex++] = s;
    }
}

//// SCFC
constexpr int per_crate_SCFC_size = 21;

void create_scfc_names_in_odb(odb crate_settings){
    int bankindex = 0;
    for(uint32_t i=0; i < N_FEBCRATES; i++){
        string feb = "Crate " + to_string(i);
        string * s = new string(feb);
        (*s) += " Index";
        crate_settings["Names SCFC"][bankindex++] = s;
        s = new string(feb);
        (*s) += " Voltage 20";
        crate_settings["Names SCFC"][bankindex++] = s;
        s = new string(feb);
        (*s) += " Voltage 3.3";
        crate_settings["Names SCFC"][bankindex++] = s;
        s = new string(feb);
        (*s) += " Voltage 5";
        crate_settings["Names SCFC"][bankindex++] = s;
        s = new string(feb);
        (*s) += " CC Temperature";
        crate_settings["Names SCFC"][bankindex++] = s;
        for(uint32_t j=0; j < MAX_FEBS_PER_CRATE; j++){
            s = new string(feb);
            (*s) += " FEB " + to_string(j) + " Temperature";
            crate_settings["Names SCFC"][bankindex++] = s;
        }
    }
}

//// SSSO
constexpr int max_sorter_inputs_per_feb = 12;
constexpr int num_sorter_counters_per_feb = 3*max_sorter_inputs_per_feb +2;
constexpr int per_fe_SSSO_size = num_sorter_counters_per_feb + 1;
const array<const string, MAX_N_SWITCHINGBOARDS> ssso = {"SCSO","SUSO","SDSO","SFSO"};
const array<const string, MAX_N_SWITCHINGBOARDS> sssonames = {"Names SCSO","Names SUSO","Names SDSO","Names SFSO"};

void create_ssso_names_in_odb(odb & settings, int switch_id){
    string sorternamestr = sssonames[switch_id];

    int bankindex = 0;
    bankindex = 0;
    for(uint32_t i=0; i < N_FEBS[switch_id]; i++){
        string feb = "FEB" + to_string(i);
        string * s = new string(feb);
        (*s) += " Index";
        settings[sorternamestr][bankindex++] = s;
        for(uint32_t j=0; j < max_sorter_inputs_per_feb; j++){
            s = new string(feb);
            (*s) += " intime hits input " + to_string(j);
            settings[sorternamestr][bankindex++] = s;
        }
        for(uint32_t j=0; j < max_sorter_inputs_per_feb; j++){
            s = new string(feb);
            (*s) += " out of time hits input " + to_string(j);
            settings[sorternamestr][bankindex++] = s;
        }
        for(uint32_t j=0; j < max_sorter_inputs_per_feb; j++){
            s = new string(feb);
            (*s) += "  overflow hits input " + to_string(j);
            settings[sorternamestr][bankindex++] = s;
        }
        s = new string(feb);
        (*s) += "  output hits";
        settings[sorternamestr][bankindex++] = s;
        s = new string(feb);
        (*s) += "  credits";
        settings[sorternamestr][bankindex++] = s;
    }
}



//// SSCN
constexpr int num_swb_counters_per_feb = 9;
const array<const string, MAX_N_SWITCHINGBOARDS> sscn = {"SCCN","SUCN","SDCN","SFCN"};
const array<const string, MAX_N_SWITCHINGBOARDS> sscnnames = {"Names SCCN","Names SUCN","Names SDCN","Names SFCN"};

void create_sscn_names_in_odb(odb & settings, int switch_id){
    string cntnamestr = sscnnames[switch_id];

    int bankindex = 0;

    settings[cntnamestr][bankindex++] = "STREAM_FIFO_FULL";
    settings[cntnamestr][bankindex++] = "BANK_BUILDER_IDLE_NOT_HEADER";
    settings[cntnamestr][bankindex++] = "BANK_BUILDER_RAM_FULL";
    settings[cntnamestr][bankindex++] = "BANK_BUILDER_TAG_FIFO_FULL";
    for(uint32_t i=0; i < N_FEBS[switch_id]; i++){
        string name = "FEB" + to_string(i);
        string * s = new string(name);
        (*s) += " Index";
        settings[cntnamestr][bankindex++] = s;
        s = new string(name);
        (*s) += " LINK FIFO ALMOST FULL";
        settings[cntnamestr][bankindex++] = s;
        s = new string(name);
        (*s) += " LINK FIFO FULL";
        settings[cntnamestr][bankindex++] = s;
        s = new string(name);
        (*s) += " SKIP EVENTS";
        settings[cntnamestr][bankindex++] = s;
        s = new string(name);
        (*s) += " NUM EVENTS";
        settings[cntnamestr][bankindex++] = s;
        s = new string(name);
        (*s) += " NUM SUB HEADERS";
        settings[cntnamestr][bankindex++] = s;
        s = new string(name);
        (*s) += " MERGER RATE";
        settings[cntnamestr][bankindex++] = s;
        s = new string(name);
        (*s) += " RESET PHASE";
        settings[cntnamestr][bankindex++] = s;
        s = new string(name);
        (*s) += " TX RESET";
        settings[cntnamestr][bankindex++] = s;
    }

}



#endif // MU3EBANKS_H
