
void sc_t::callback(alt_u16 cmd, volatile alt_u32* data, alt_u16 n) {
    auto& regs = ram->regs.scifi;
    const int n_ASICS=4;
    switch(cmd){
    case 0x0101: //power up (not implemented in current FEB)
        break;
    case 0x0102: //power down (not implemented in current FEB)
        break;
    case 0x0103: //configure all off
	printf("[scifi] configuring all off\n");
        //for(int i=0;i<scifi_module::n_ASICS;i++)
        //    scifi_module.configure_asic(mutrig_config_ALL_OFF);
        break;
    case 0xfffe:
	printf("-ping-\n");
        break;
    case 0xffff:
        break;
    default:
        if((cmd&0xfff0) ==0x0110){ //configure ASIC
		uint8_t chip=cmd&0x000f;
		scifi_module.configure_asic(chip,(alt_u8*) data);
        }
    }
}
