#include "mu3ebanks.h"



void mu3ebanks::create_ssfe_names_in_odb(odb & settings, int switch_id){
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
void mu3ebanks::create_scfc_names_in_odb(odb crate_settings){
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
void mu3ebanks::create_ssso_names_in_odb(odb & settings, int switch_id){
    string sorternamestr = sssonames[switch_id];

    int bankindex = 0;
    bankindex = 0;
    for(uint32_t i=0; i < N_FEBS[switch_id]; i++){
        string feb = "FEB" + to_string(i);
        string * s = new string(feb);
        (*s) += " Sorter Index";
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
void mu3ebanks::create_sscn_names_in_odb(odb & settings, int switch_id){
    string cntnamestr = sscnnames[switch_id];

    int bankindex = 0;

    settings[cntnamestr][bankindex++] = "STREAM_FIFO_FULL";
    settings[cntnamestr][bankindex++] = "BANK_BUILDER_IDLE_NOT_HEADER";
    settings[cntnamestr][bankindex++] = "BANK_BUILDER_RAM_FULL";
    settings[cntnamestr][bankindex++] = "BANK_BUILDER_TAG_FIFO_FULL";
    for(uint32_t i=0; i < N_FEBS[switch_id]; i++){
        string name = "FEB" + to_string(i);
        string * s = new string(name);
        (*s) += " SWB CNT Index";
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

void mu3ebanks::create_sspl_names_in_odb(odb & settings, int switch_id){
    string cntnamestr = ssplnames[switch_id];

    int bankindex = 0;

    settings[cntnamestr][bankindex++] = "PLL156";
    settings[cntnamestr][bankindex++] = "PLL250";
    settings[cntnamestr][bankindex++] = "Links Locked Low";
    settings[cntnamestr][bankindex++] = "Links Locked High";
}


void mu3ebanks::create_psls_names_in_odb(odb & settings, int switch_id, uint32_t n_febs_mupix){
    int bankindex = 0;
    uint32_t nlinks = MAX_LVDS_LINKS_PER_FEB;
    string cntnamestr = pslsnames[switch_id];

    for(uint32_t i=0; i < n_febs_mupix; i++){
        string name = "FEB" + to_string(i);
        settings[cntnamestr][bankindex++] = name;
        string * s = new string(name);
            (*s) += " N LVDS Links";
        settings[cntnamestr][bankindex++] = s;
        
        for(uint32_t j=0; j < nlinks; j++){
            string name = "F" + to_string(i) + "L" + to_string(j);
            string * s = new string(name);
            (*s) += " Status";
            settings[cntnamestr][bankindex++] = s;
            s = new string(name);
            (*s) += " Num Hits LVDS";
            settings[cntnamestr][bankindex++] = s;
            s = new string(name);
            (*s) += " Arrival Histogram Bin 0";
            settings[cntnamestr][bankindex++] = s;
            s = new string(name);
            (*s) += " Arrival Histogram Bin 1";
            settings[cntnamestr][bankindex++] = s;
             s = new string(name);
            (*s) += " Arrival Histogram Bin 2";
            settings[cntnamestr][bankindex++] = s;
            s = new string(name);
            (*s) += " Arrival Histogram Bin 3";
            settings[cntnamestr][bankindex++] = s;                      
        }
    }
}

