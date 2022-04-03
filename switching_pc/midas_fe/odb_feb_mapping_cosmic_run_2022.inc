odb cr("/Equipment/Clock Reset/Settings");
cr["SwitchingBoardMask"][0] = 1;
cr["SwitchingBoardMask"][1] = 0;
cr["SwitchingBoardMask"][2] = 0;
cr["SwitchingBoardMask"][3] = 0;


odb crates("/Equipment/FEBCrates/Settings");
// TODO: MSCB names here
crates["CrateControllerMSCB"][0] = "mscb420";
crates["CrateControllerMSCB"][1] = "mscb421";
crates["CrateControllerMSCB"][2] = "mscb425";
crates["CrateControllerMSCB"][3] = "mscb426";
// Below: Leave empty for the integration run
crates["CrateControllerMSCB"][4] = "";
crates["CrateControllerMSCB"][5] = "";
crates["CrateControllerMSCB"][6] = "";
crates["CrateControllerMSCB"][7] = "";

// TODO: MSCB nodes here
crates["CrateControllerNode"][0] = 1;
crates["CrateControllerNode"][1] = 1;
crates["CrateControllerNode"][2] = 1;
crates["CrateControllerNode"][3] = 1;
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

for(int i=12; i< 128; i++){
    crates["FEBCrate"][i] = 255;
    crates["FEBSlot"][i] = 255;
}


odb links("/Equipment/LinksCentral/Settings");

for(uint32_t i=0; i < 12; i++){
    links["LinkMask"][i] = FEBLINKMASK::ON;
}

for(uint32_t i=12; i < 14; i++){
    links["LinkMask"][i] = FEBLINKMASK::DataOn;
}

for(uint32_t i=14; i < 34; i++){
    links["LinkMask"][i] = FEBLINKMASK::OFF;
}


for(uint32_t i=0; i < 12; i++){
    links["LinkFEB"][i] = i;
}

for(uint32_t i=12; i < 14; i++){
    links["LinkFEB"][i] = i-2;
}

for(uint32_t i=14; i < 34; i++){
    links["LinkFEB"][i] = 255;
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

for(uint32_t i=14; i < 34; i++){
    links["FrontEndBoardType"][i] = FEBTYPE::Unused;
}

links["FrontEndBoardNames"][0] = "Pixel US L1 0-3";
links["FrontEndBoardNames"][1] = "Pixel US L1 4-7";
links["FrontEndBoardNames"][2] = "Pixel US L2 0-3";
links["FrontEndBoardNames"][3] = "Pixel US L2 4-7";
links["FrontEndBoardNames"][4] = "Pixel US L2 8-9";
links["FrontEndBoardNames"][5] = "Pixel DS L1 0-3";
links["FrontEndBoardNames"][6] = "Pixel DS L1 4-7";
links["FrontEndBoardNames"][7] = "Pixel DS L2 0-3";
links["FrontEndBoardNames"][8] = "Pixel DS L2 4-7";
links["FrontEndBoardNames"][9] = "Pixel DS L2 8-9";
links["FrontEndBoardNames"][10] = "Fibre US";
links["FrontEndBoardNames"][11] = "Fibre DS";
links["FrontEndBoardNames"][12] = "Fibre US secondray";
links["FrontEndBoardNames"][13] = "Fibre DS secondary";


