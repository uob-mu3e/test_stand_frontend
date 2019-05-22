/*
 * xcvr.h
 *
 *  Created on: Nov 10, 2017
 *      Author: akozlins
 */

#ifndef SOFTWARE_APP_SRC_XCVR_H_
#define SOFTWARE_APP_SRC_XCVR_H_

#include "system.h"

struct qsfp_t {
    volatile alt_u32* phy_base[4] = {
        (alt_u32*)(QSFPA_XCVR_RECONFIG_PHY_BASE | ALT_CPU_DCACHE_BYPASS_MASK),
        (alt_u32*)(0),
        (alt_u32*)(QSFPC_XCVR_RECONFIG_PHY_BASE | ALT_CPU_DCACHE_BYPASS_MASK),
        (alt_u32*)(QSFPD_XCVR_RECONFIG_PHY_BASE | ALT_CPU_DCACHE_BYPASS_MASK),
    };

    volatile alt_u32* pll_base[4] = {
        (alt_u32*)(QSFPA_XCVR_RECONFIG_PLL_BASE | ALT_CPU_DCACHE_BYPASS_MASK),
        (alt_u32*)(0),
        (alt_u32*)(QSFPC_XCVR_RECONFIG_PLL_BASE | ALT_CPU_DCACHE_BYPASS_MASK),
        (alt_u32*)(QSFPD_XCVR_RECONFIG_PLL_BASE | ALT_CPU_DCACHE_BYPASS_MASK),
    };

    volatile alt_u32* mm_base[4] = {
        (alt_u32*)(QSFPA_XCVR_MM_BASE | ALT_CPU_DCACHE_BYPASS_MASK),
        (alt_u32*)(0),
        (alt_u32*)(QSFPC_XCVR_MM_BASE | ALT_CPU_DCACHE_BYPASS_MASK),
        (alt_u32*)(QSFPD_XCVR_MM_BASE | ALT_CPU_DCACHE_BYPASS_MASK),
    };

    volatile alt_u32* bert_base[4] = {
        (alt_u32*)(QSFPA_BERT_0_BASE | ALT_CPU_DCACHE_BYPASS_MASK),
        (alt_u32*)(0),
        (alt_u32*)(QSFPC_BERT_0_BASE | ALT_CPU_DCACHE_BYPASS_MASK),
        (alt_u32*)(QSFPD_BERT_0_BASE | ALT_CPU_DCACHE_BYPASS_MASK),
    };

    alt_u32 link = 0;

    void init(alt_u32 link_) {
        if(link_ > 3 || link_ == 1) return; // FIXME:
        link = link_;
    }

    volatile alt_u32* phy(int ch) {
        return phy_base[link] + ch * 0x1000 / 4;
    }

    volatile alt_u32* fpll() {
        return pll_base[link];
    }

    void print() {
        volatile alt_u32* mm = mm_base[link];
        alt_u32 label = ((alt_u32)mm >> 20) & 0xF;
        printf("           SLIP SDR SYNC DISP\n");
        printf("QSFP[%1X] => %04x %03X %04X %04X\n", label, mm[0] >> 16, mm[0] & 0xFFF, mm[1] >> 16, mm[1] & 0xFFFF
        );
    }

    volatile alt_u32* bert(int ch) {
        return bert_base[link] + ch * 0x100 / 4;
    }
} qsfp;

struct reconfig_t {
    static const alt_u32 ONE = 1;

    volatile alt_u32* base;

    /**
     * Set bit 'i'.
     */
    void set(alt_u32 r, alt_u32 i, alt_u32 b) {
        base[r] ^= (-b ^ base[r]) & (ONE << i);
        printf("[0x%03X](%u) <= %u => [0x%03X] is 0x%02X\n", r, i, b, r, base[r]);
    }

    /**
     * Get bit i;
     */
    alt_u32 get(alt_u32 r, alt_u32 i) {
        alt_u32 b = (base[r] >> i) & ONE;
        printf("[0x%03X](%u) is '%u'\n", r, i, b);
        return b;
    }

    void fpll() {
        printf("reconfig fpll\n");
        base = qsfp.fpll();

        if(get(0x280, 0) == 1) return; // pll_locked

        // 1. Request user access to the internal configuration bus by writing 0x2 to offset address 0x0[7:0].
        base[0x000] = 0x2;
        // 2. Wait for reconfig_waitrequest to be deasserted (logic low)
        //    or wait until capability register of PreSICE Avalon-MM interface control = 0x0.
        while(get(0x280, 2) != 0) usleep(100);
        // 3. To calibrate the fPLL, Read-Modify-Write 0x1 to bit[1] of address 0x100 of the fPLL.
        set(0x100, 1, 1);
        // 4. Release the internal configuration bus to PreSICE to perform recalibration by writing 0x1 to offset address 0x0[7:0].
        base[0x000] = 0x1;
        // 5. Periodically check the *cal_busy output signals
        //    or read the capability registers 0x280[1] to check *cal_busy status until calibration is complete.
        while(get(0x280, 1) == 1) usleep(100); // fPLL calibration is running

        while(get(0x280, 0) != 1) usleep(100); // wait pll_locked
    }

    void phy(alt_u32 ch) {
        printf("reconfig phy %u\n", ch);
        base = qsfp.phy(ch);

        // 1. Request access to the internal configuration bus by writing 0x2 to offset address 0x0[7:0].
        base[0x000] = 0x2;
        // 2. Wait for reconfig_waitrequest to be deasserted (logic low),
        // or wait until capability register of PreSICE Avalon-MM interface control = 0x0.
        while(get(0x281, 2) != 0) usleep(100);
        // 3. Set the proper value to offset address 0x100 to enable PMA calibration.
        //    You must set 0x100[6] to 0x0 when you enable any calibration.
        set(0x100, 1, 1); // PMA RX calibration enable
        set(0x100, 5, 1); // PMA TX calibration enable
        set(0x100, 6, 0);
        // 4. Set the rate switch flag register for PMA RX calibration after the rate change.
        //    - Read-Modify-Write 0x1 to offset address 0x166[7] if no rate switch.
        //    - Read-Modify-Write 0x0 to offset address 0x166[7] if switched rate with different CDR bandwidth setting.
        set(0x166, 7, 1);
        // 5. Do Read-Modify-Write the proper value to capability register 0x281[5:4] for PMA calibration to enable/disable tx_cal_busy or rx_cal_busy output.
        //    - To enable rx_cal_busy, Read-Modify-Write 0x1 to 0x281[5].
        //    - To disable rx_cal_busy, Read-Modify-Write 0x0 to 0x281[5].
        //    - To enable tx_cal_busy, Read-Modify-Write 0x1 to 0x281[4].
        //    - To disable tx_cal_busy, Read-Modify-Write 0x0 to 0x281[4].
        set(0x281, 4, 1); // enable PMA channel tx_cal_busy output
        set(0x281, 5, 1); // enable PMA channel rx_cal_busy output
        // 6. Release the internal configuration bus to PreSICE to perform recalibration by writing 0x1 to offset address 0x0[7:0].
        base[0x000] = 0x1;
        // 7. Periodically check the *cal_busy output signals
        //    or read the capability registers 0x281[1:0] to check *cal_busy status until calibration is complete.
        while(get(0x281, 0) == 1) usleep(100); // PMA TX calibration is running
        while(get(0x281, 1) == 1) usleep(100); // PMA RX calibration is running
    }
};

struct bert_t {
    alt_u32 tx_[16];
    alt_u32 rx_[16];
    alt_u32 __reserved[32];

    static alt_u32 lfsr(alt_u32 r) {
        alt_u32 f = (r >> 31) xor (r >> 21) xor (r >> 1) xor (r >> 0);
        return (r << 1) | (f & 1);
    }
};

void menu_xcvr() {
    static reconfig_t reconfig;

    int link = 0, ch = 0;

    while (1) {
        qsfp.init(link);
        volatile bert_t* berts = (bert_t*)qsfp.bert(0);

        char cmd;
        if(read(uart, &cmd, 1) > 0) switch(cmd) {
        case '[':
            if(link > 0) link--;
            break;
        case ']':
            if(link < 3) link++;
            break;
        case '0': case '1': case '2': case '3': // select channel
            ch = cmd - '0';
            break;
        case 'R': // reconfig all
            reconfig.fpll();
            for(int i = 0; i < 4; i++) reconfig.phy(i);
            break;
        case 'r': // reconfig ch
            reconfig.phy(ch);
            break;
        case 'I': // init all
            for(int i = 0; i < 4; i++) berts[i].tx_[0] = 0x00000001;
            break;
        case 'i': // init ch
            berts[ch].tx_[0] = 0x00000001;
            break;
        case 'E': // inject error
            for(int i = 0; i < 4; i++) berts[i].tx_[4] = 0x00000001;
            break;
        case 'e': // inject error
            berts[ch].tx_[4] = 0x00000001;
            break;
        case 'C':
            for(int i = 0; i < 4; i++) berts[i].tx_[1] = 0x00000000;
            break;
        case 'c':
            berts[ch].tx_[1] = 0x00000000;
            break;
        case '?':
            wait_key();
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }

        qsfp.print();

        auto& rx_ = berts[ch].rx_;
        alt_u32 cnt = rx_[0x8];
        alt_u32 ben = rx_[0x4];
        printf("[%d] %u Gbit => %u\n", ch, cnt, ben);
        alt_u32 lfsr0 = rx_[0x9], lfsr1 = bert_t::lfsr(lfsr0), lfsr2 = bert_t::lfsr(lfsr1);
        alt_u32 data0 = rx_[0xA];
        printf("    data0 = 0x%08X 0x%08X\n", data0, lfsr0);
//        printf("    data1 = 0x%08X 0x%08X\n", rx.data1, lfsr1);
//        printf("    data2 = 0x%08X 0x%08X\n", rx.data2, lfsr2);
        printf("\n");

        usleep(200000);
    }
}

#endif /* SOFTWARE_APP_SRC_XCVR_H_ */
