struct mupix_datapath_t {
    sc_t* sc;
    mupix_datapath_t(sc_t* sc_): sc(sc_){};
    
    void menu() {
        auto& regs = sc->ram->regs.scifi;
        alt_u32 datagenreg = 0x0;
        char str[2] = {0};
        
        while(1) {
            printf("  [0] => write datagen control reg\n");
            printf("  [1] => read  datagen control reg\n");
            
            if((sc->ram->data[0xFF65] >> 31) & 1U){
                printf("  [2] => disable data gen\n");
            }else{
                printf("  [2] => enable data gen\n");
            }
            if((sc->ram->data[0xFF65] >> 16) & 1U){
                printf("  [3] => disengage data gen\n");
            }else{
                printf("  [3] => engage data gen\n");
            }
            if((sc->ram->data[0xFF65] >> 17) & 1U){
                printf("  [4] => insert before sorter\n");
            }else{
                printf("  [4] => insert after sorter\n");
            }
            if((sc->ram->data[0xFF65] >> 4) & 1U){
                printf("\n RATE: Full Stream\n");
            }else{
                printf("\n RATE:0x%01x\n", 0xF - (sc->ram->data[0xFF65] & 0xF) + 1);
            }
            printf("  [+]\n");
            printf("  [-]\n");
            
            printf("Select entry ...\n");
            char cmd = wait_key();
            switch(cmd) {
            case '0':
                datagenreg = 0x0;
                printf("Enter datagen reg in hex: ");

                for(int i = 0; i<8; i++){
                    printf("mask: 0x%08x\n", datagenreg);
                    str[0] = wait_key();
                    datagenreg = datagenreg*16+strtol(str,NULL,16);
                }

                printf("setting reg to 0x%08x\n", datagenreg);
                sc->ram->data[0xFF65] = datagenreg;
                break;
            case '1':
                printf("0x%08x\n", sc->ram->data[0xFF65]);
                break;
            case '2':
                sc->ram->data[0xFF65] ^= 1UL << 31;
                break;
            case '3':
                sc->ram->data[0xFF65] ^= 1UL << 16;
                break;
            case '4':
                sc->ram->data[0xFF65] ^= 1UL << 17;
            case '-':
                if((sc->ram->data[0xFF65] >> 4) & 1U){
                    sc->ram->data[0xFF65] ^= 1UL << 4;
                }else if(sc->ram->data[0xFF65] & 1U and (sc->ram->data[0xFF65] >> 1) & 1U and (sc->ram->data[0xFF65] >> 2) & 1U and (sc->ram->data[0xFF65] >> 3) & 1U){

                }else{
                    sc->ram->data[0xFF65] = sc->ram->data[0xFF65] + 1;
                }
                break;
            case '+':
                if(!((sc->ram->data[0xFF65]) & 1U) and !((sc->ram->data[0xFF65] >> 1) & 1U) and !((sc->ram->data[0xFF65] >> 2) & 1U) and !((sc->ram->data[0xFF65] >> 3) & 1U)){
                    sc->ram->data[0xFF65] |= 1UL << 4;
                }else{
                    sc->ram->data[0xFF65] = sc->ram->data[0xFF65] - 1;
                }
                break;
            case 'q':
                return;
            default:
                printf("invalid command: '%c'\n", cmd);
            }
        }
    }


};
