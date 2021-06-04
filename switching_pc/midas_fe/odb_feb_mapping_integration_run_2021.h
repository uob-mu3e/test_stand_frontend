odb links("/Equipment/Links/Settings");
links["SwitchingBoardMask"][0] = 1;
links["SwitchingBoardMask"][1] = 0;
links["SwitchingBoardMask"][2] = 0;
links["SwitchingBoardMask"][3] = 0;

for(uint32_t i=0; i < 12; i++){
    links["LinkMask"][i] = FEBLINKMASK::ON;
}

for(uint32_t i=12; i < 14; i++){
    links["LinkMask"][i] = FEBLINKMASK::DataOn;
}

for(uint32_t i=14; i < 192; i++){
    links["LinkMask"][i] = FEBLINKMASK::OFF;
}

for(uint32_t i=0; i < 10; i++){
    links["FrontEndBoardType"][i] = FEBTYPE::Pixel;
}

for(uint32_t i=10; i < 12; i++){
    links["FrontEndBoardType"][i] = FEBTYPE::Fibre;
}

for(uint32_t i=12; i < 14; i++){
    links["FrontEndBoardType"][i] = FEBTYPE::FibreSecondary;
}

for(uint32_t i=14; i < 192; i++){
    links["FrontEndBoardType"][i] = FEBTYPE::Undefined;
}

links["FrontEndBoardNames"][0] = "Pixel US L1 0-3";
links["FrontEndBoardNames"][1] = "Pixel US L1 4-7";
links["FrontEndBoardNames"][2] = "Pixel US L2 0-3";
links["FrontEndBoardNames"][3] = "Pixel US L2 4-6";
links["FrontEndBoardNames"][4] = "Pixel US L2 7-9";
links["FrontEndBoardNames"][5] = "Pixel DS L1 0-3";
links["FrontEndBoardNames"][6] = "Pixel DS L1 4-7";
links["FrontEndBoardNames"][7] = "Pixel DS L2 0-3";
links["FrontEndBoardNames"][8] = "Pixel DS L2 4-6";
links["FrontEndBoardNames"][9] = "Pixel DS L2 7-9";
links["FrontEndBoardNames"][11] = "Fibre US";
links["FrontEndBoardNames"][12] = "Fibre DS";
links["FrontEndBoardNames"][13] = "Fibre US secondray";
links["FrontEndBoardNames"][14] = "Fibre DS secondary";

odb crates("/Equipment/FEBCrates/Settings");
// TODO: MSCB names here
crates["CrateControllerMSCB"][0] = "";
crates["CrateControllerMSCB"][1] = "";
crates["CrateControllerMSCB"][2] = "";
crates["CrateControllerMSCB"][3] = "";
// Below: Leave empty for the integration run
crates["CrateControllerMSCB"][4] = "";
crates["CrateControllerMSCB"][5] = "";
crates["CrateControllerMSCB"][6] = "";
crates["CrateControllerMSCB"][7] = "";

// TODO: MSCB nodes here
crates["CrateControllerNode"][0] = 0;
crates["CrateControllerNode"][1] = 0;
crates["CrateControllerNode"][2] = 0;
crates["CrateControllerNode"][3] = 0;
// Below: Leave empty for the integration run
crates["CrateControllerNode"][4] = 0;
crates["CrateControllerNode"][5] = 0;
crates["CrateControllerNode"][6] = 0;
crates["CrateControllerNode"][7] = 0;

crates["FEBCrate"][0] = 1-1;
crates["FEBSlot"][0] = 0;
crates["FEBCrate"][1] = 1-1;
crates["FEBSlot"][1] = 1;
crates["FEBCrate"][2] = 1-1;
crates["FEBSlot"][2] = 2;
crates["FEBCrate"][3] = 1-1;
crates["FEBSlot"][3] =  3;
crates["FEBCrate"][4] = 2-1;
crates["FEBSlot"][4] =  0;

crates["FEBCrate"][5] = 3-1;
crates["FEBSlot"][5] = 3;
crates["FEBCrate"][6] = 3-1;
crates["FEBSlot"][6] = 3;
crates["FEBCrate"][7] = 3-1;
crates["FEBSlot"][7] = 1;
crates["FEBCrate"][8] = 3-1;
crates["FEBSlot"][8] =  0;
crates["FEBCrate"][9] = 4-1;
crates["FEBSlot"][9] =  3;

crates["FEBCrate"][10] = 2-1;
crates["FEBSlot"][10] = 3;
crates["FEBCrate"][11] = 4-1;
crates["FEBSlot"][11] = 0;

for(int i=12; i< 192; i++){
    crates["FEBCrate"][i] = 255;
    crates["FEBSlot"][i] = 255;
}
