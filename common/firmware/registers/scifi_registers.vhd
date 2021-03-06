-- Register Map
-- Note: 
-- write register, use naming scheme:       ***_REGISTER_W
-- read  register, use naming scheme:       ***_REGISTER_R
-- bit range     , use naming scheme:       ***_RANGE
-- single bit constant, use naming scheme:  ***_BIT

-- M.Mueller, L.Gerritzen, May 2021

library ieee;
use ieee.std_logic_1164.all;

package scifi_registers is

--////////////////////////////////////////////--
--//////////////////REGISTER MAP//////////////--
--////////////////////////////////////////////--

-----------------------------------------------------------------
---- scifi_registers---------------------------------------------
-----------------------------------------------------------------
    -- counters
    constant SCIFI_CNT_CTRL_REGISTER_W          :   integer := 16#4040#;
    constant SCIFI_CNT_VALUE_REGISTER_R         :   integer := 16#4041#;
    constant SCIFI_CNT_ADDR_REGISTER_W          :   integer := 16#4042#;
    constant SCIFI_CNT_MISO_TRANSITION_REGISTER_R :   integer := 16#4043#;

    -- monitors
    constant SCIFI_MON_STATUS_REGISTER_R        :   integer := 16#4044#;
    constant SCIFI_MON_RX_DPA_LOCK_REGISTER_R   :   integer := 16#4045#;
    constant SCIFI_MON_RX_READY_REGISTER_R      :   integer := 16#4046#;

    -- ctrl
    constant SCIFI_CTRL_DUMMY_REGISTER_W        :   integer := 16#4047#;
    -- TODO: Name single bits according to this:
    --        printf("dummyctrl_reg:    0x%08X\n", regs.ctrl.dummy);
    --        printf("    :cfgdummy_en  0x%X\n", (regs.ctrl.dummy>>0)&1);
    --        printf("    :datagen_en   0x%X\n", (regs.ctrl.dummy>>1)&1);
    --        printf("    :datagen_fast 0x%X\n", (regs.ctrl.dummy>>2)&1);
    --        printf("    :datagen_cnt  0x%X\n", (regs.ctrl.dummy>>3)&0x3ff);
    constant SCIFI_CTRL_DP_REGISTER_W            :   integer := 16#4048#;
    constant SCIFI_CTRL_RESET_REGISTER_W         :   integer := 16#4049#;
    constant SCIFI_CTRL_RESETDELAY_REGISTER_W    :   integer := 16#404A#;

    -- lapse counting
    constant SCIFI_CTRL_LAPSE_COUNTER_REGISTER_W :   integer := 16#404B#;

    -- cc diff
    constant SCIFI_CC_DIFF_REGISTER_R            :   integer := 16#404C#;
    
    -- link data debug path
    constant SCIFI_LINK_DATA_REGISTER_W          :   integer := 16#404D#;

    -- ch rate register (times 128 at the moment)
    constant SCIFI_CH_RATE_REGISTER_R            :   integer := 16#404E#;

end package;
