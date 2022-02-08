odb links("/Equipment/Links/Settings");
links["SwitchingBoardMask"][0] = 1;
links["SwitchingBoardMask"][1] = 0;
links["SwitchingBoardMask"][2] = 0;
links["SwitchingBoardMask"][3] = 0;

for(uint32_t i=0; i < 2; i++){
    links["LinkMask"][i] = FEBLINKMASK::ON;
}

for(uint32_t i=2; i < 192; i++){
    links["LinkMask"][i] = FEBLINKMASK::OFF;
}

for(uint32_t i=0; i < 2; i++){
    links["FrontEndBoardType"][i] = FEBTYPE::Pixel;
}

for(uint32_t i=2; i < 192; i++){
    links["FrontEndBoardType"][i] = FEBTYPE::Unused;
}

links["FrontEndBoardNames"][0] = "Pixel US";
links["FrontEndBoardNames"][1] = "Pixel DS";
