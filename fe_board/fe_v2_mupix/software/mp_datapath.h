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
            printf("  [5] => print sorter counters\n");
            printf("  [6] => set data bypass select\n");
            printf("  [7] => write sorter inject reg\n");
            printf("  [8] => print hit ena counters\n");
            
            if((sc->ram->data[0xFF65] >> 4) & 1U){
                printf("\n DataGen RATE: Full Stream\n");
            }else{
                printf("\n DataGen RATE:0x%01x\n", 0xF - (sc->ram->data[0xFF65] & 0xF) + 1);
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
            case '5':
                printf("sorter counters:\n");
                for(int i=0; i<41; i++){
                    printf("%i: 0x%08x\n",i,sc->ram->data[0xFF92+i]);
                }
                break;
            case '6':
                datagenreg = 0x0;
                printf("enter data bypass select in hex: ");

                for(int i = 0; i<8; i++){
                    printf("mask: 0x%08x\n", datagenreg);
                    str[0] = wait_key();
                    datagenreg = datagenreg*16+strtol(str,NULL,16);
                }

                printf("setting reg to 0x%08x\n", datagenreg);
                sc->ram->data[0xFFBB] = datagenreg;
                break;
            case '7':
                datagenreg = 0x0;
                sc->ram->data[0xFFBE] = datagenreg;
                printf("enter sorter inject reg in hex: ");

                for(int i = 0; i<8; i++){
                    printf("mask: 0x%08x\n", datagenreg);
                    str[0] = wait_key();
                    datagenreg = datagenreg*16+strtol(str,NULL,16);
                }
                printf("setting reg to 0x%08x\n", datagenreg);
                
                for(int i=0; i<10000; i++){
                    sc->ram->data[0xFFBE] = datagenreg;
                    sc->ram->data[0xFFBE] = 0x0;
                    usleep(50);
                }
                break;
            case '8':
                printf("hit ena counters:\n");
                sc->ram->data[0xFFC0] = 0x0;
                for(int i=0; i<40; i++){
                    printf("%i: 0x%08x\n",i,sc->ram->data[0xFFBF]);
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
