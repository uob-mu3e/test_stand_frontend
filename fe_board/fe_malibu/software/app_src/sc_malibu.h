
alt_u16 sc_t::callback(alt_u16 cmd, volatile alt_u32* data, alt_u16 n) {
    switch(cmd) {
    case 0x0101:
        malibu.power_TMB(true);
        break;
    case 0x0102:
        malibu.power_TMB(false);
        break;
    case 0x0103:
        malibu.chip_configure(0, stic3_config_ALL_OFF);
        break;
    case 0x0104:
        malibu.chip_configure(0, stic3_config_PLL_TEST_ch0to6_noGenIDLE);
        break;
    case 0x0105:
        malibu.i2c_write_u32(data, n);
        break;

    case 0xFFFF:
        break;

    default:
        if((cmd & 0xFFF0) == 0x0110) {
            printf("try stic_configure\n");
            int stic = cmd & 0x000F;
            int ok = 1;
            for(int i = 0; i < sizeof(stic3_config_ALL_OFF) / sizeof(stic3_config_ALL_OFF[0]); i++) {
                alt_u8 b = ((alt_u8*)data)[i];
                if(b != stic3_config_ALL_OFF[i]) ok = 0;
            }
            printf("ok = %d\n", ok);
            if(ok == 1) malibu.chip_configure(stic, (alt_u8*)data);
        }
        else {
            printf("[sc_callback] unknown command\n");
        }
    }

    return 0;
}
