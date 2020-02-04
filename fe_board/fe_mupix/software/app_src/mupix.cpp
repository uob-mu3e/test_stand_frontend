#include "mupix.h"
#include "default_mupix_dacs.h"
char wait_key(useconds_t us = 100000);


#include "../../../fe/software/app_src/sc.h"

#include <ctype.h>

//configure ASIC
alt_u16 mupix_t::set_chip_dacs(alt_u32 asic, volatile alt_u32* bitpattern) {
    printf("[mupix] configure asic(%u)\n", asic);

    sc->ram->data[0xFF8D] = 0x005e0000 + (5 << asic); // 4 Sensors
    for(int i = 0; i < MUPIX8_LEN32; i++) {
        sc->ram->data[0xFF8D] = bitpattern[i];
        //printf("0x%08x\n",bitpattern[i]);
        }
   sc->ram->data[0xFF8E] = 0x00100001;
   sc->ram->data[0xFF95] = 0;
 
    return FEB_REPLY_SUCCESS;
}

//configure Board
alt_u16 mupix_t::set_board_dacs(alt_u32 asic, volatile alt_u32* bitpattern) {
    printf("[mupix] configure board(%u)\n", asic);

    for(unsigned int i = 0; i < MUPIXBOARD_LEN32; i++) {
        sc->ram->data[0xFF83+i] = bitpattern[i];
        //printf("0x%08x\n",bitpattern[i]);
        }
   sc->ram->data[0xFF8C] = 0x1;
   sc->ram->data[0xFF8C] = 0x0;
 
    return FEB_REPLY_SUCCESS;
}

extern int uart;
void mupix_t::menu(){

    auto& regs = sc->ram->regs.scifi;
    while(1) {
        printf("  [b] => set default board DACs (All)\n");
        printf("  [0] => set default chip A DACs\n");
        printf("  [1] => set default chip B DACs\n");
        printf("  [2] => set default chip C DACs\n");
        printf("  [3] => set default chip E DACs\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '0':
            set_chip_dacs(0, default_mupix_dacs);
            break;
        case '1':
            set_chip_dacs(1, default_mupix_dacs);
            break;
        case '2':
            set_chip_dacs(2, default_mupix_dacs);
            break;
        case '3':
            set_chip_dacs(3, default_mupix_dacs);
            break;
        case 'b':
            set_board_dacs(0, default_board_dacs);
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}



alt_u16 mupix_t::callback(alt_u16 cmd, volatile alt_u32* data, alt_u16 n) {
//    auto& regs = ram->regs.scifi;
    alt_u16 status=FEB_REPLY_SUCCESS; 
    switch(cmd){
    case 0x0101: //power up (not implemented in current FEB)
        break;
    case 0x0102: //power down (not implemented in current FEB)
        break;
    case 0xfffe:
	printf("-ping-\n");
        break;
    case 0xffff:
        break;
    case 0x0110:
           status=set_chip_dacs(data[0], &(data[1]));
           return status;
    case 0x0120:
           status=set_board_dacs(data[0], &(data[1]));

/*
        if(sc->ram->regs.scifi.ctrl.dummy&1){
              //when configured as dummy do the spi transaction,
              //but always return success to switching board
	      if(status!=FEB_REPLY_SUCCESS) printf("[WARNING] Using configuration dummy\n");
              status=FEB_REPLY_SUCCESS;

           }*/
	   return status;
    default:
        return FEB_REPLY_ERROR;
    }

    return 0;
}
