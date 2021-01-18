---------------------------------------
--
-- On detector FPGA for layer 0 - transceiver component library
-- Sebastian Dittmeier, July 2016
-- 
-- dittmeier@physi.uni-heidelberg.de
--
----------------------------------



library ieee;
use ieee.std_logic_1164.all;

package transceiver_components is

component ip_altera_xcvr_native_av
	PORT
	(
		pll_powerdown           : in  std_logic_vector(3 downto 0)   := (others => '0'); --           pll_powerdown.pll_powerdown
		tx_analogreset          : in  std_logic_vector(3 downto 0)   := (others => '0'); --          tx_analogreset.tx_analogreset
		tx_digitalreset         : in  std_logic_vector(3 downto 0)   := (others => '0'); --         tx_digitalreset.tx_digitalreset
		tx_pll_refclk           : in  std_logic_vector(0 downto 0)   := (others => '0'); --           tx_pll_refclk.tx_pll_refclk
		tx_serial_data          : out std_logic_vector(3 downto 0);                      --          tx_serial_data.tx_serial_data
		pll_locked              : out std_logic_vector(3 downto 0);                      --              pll_locked.pll_locked
		rx_analogreset          : in  std_logic_vector(3 downto 0)   := (others => '0'); --          rx_analogreset.rx_analogreset
		rx_digitalreset         : in  std_logic_vector(3 downto 0)   := (others => '0'); --         rx_digitalreset.rx_digitalreset
		rx_cdr_refclk           : in  std_logic_vector(0 downto 0)   := (others => '0'); --           rx_cdr_refclk.rx_cdr_refclk
		rx_serial_data          : in  std_logic_vector(3 downto 0)   := (others => '0'); --          rx_serial_data.rx_serial_data
		rx_is_lockedtoref       : out std_logic_vector(3 downto 0);                      --       rx_is_lockedtoref.rx_is_lockedtoref
		rx_is_lockedtodata      : out std_logic_vector(3 downto 0);                      --      rx_is_lockedtodata.rx_is_lockedtodata
		rx_seriallpbken         : in  std_logic_vector(3 downto 0)   := (others => '0'); --         rx_seriallpbken.rx_seriallpbken
		tx_std_coreclkin        : in  std_logic_vector(3 downto 0)   := (others => '0'); --        tx_std_coreclkin.tx_std_coreclkin
		rx_std_coreclkin        : in  std_logic_vector(3 downto 0)   := (others => '0'); --        rx_std_coreclkin.rx_std_coreclkin
		tx_std_clkout           : out std_logic_vector(3 downto 0);                      --           tx_std_clkout.tx_std_clkout
		rx_std_clkout           : out std_logic_vector(3 downto 0);                      --           rx_std_clkout.rx_std_clkout
		rx_std_byteorder_ena    : in  std_logic_vector(3 downto 0)   := (others => '0'); --    rx_std_byteorder_ena.rx_std_byteorder_ena
		rx_std_byteorder_flag   : out std_logic_vector(3 downto 0);                      --   rx_std_byteorder_flag.rx_std_byteorder_flag
		rx_std_wa_patternalign  : in  std_logic_vector(3 downto 0)   := (others => '0'); --  rx_std_wa_patternalign.rx_std_wa_patternalign
		tx_cal_busy             : out std_logic_vector(3 downto 0);                      --             tx_cal_busy.tx_cal_busy
		rx_cal_busy             : out std_logic_vector(3 downto 0);                      --             rx_cal_busy.rx_cal_busy
		reconfig_to_xcvr        : in  std_logic_vector(559 downto 0) := (others => '0'); --        reconfig_to_xcvr.reconfig_to_xcvr
		reconfig_from_xcvr      : out std_logic_vector(367 downto 0);                    --      reconfig_from_xcvr.reconfig_from_xcvr
		tx_parallel_data        : in  std_logic_vector(127 downto 0) := (others => '0'); --        tx_parallel_data.tx_parallel_data
		tx_datak                : in  std_logic_vector(15 downto 0)  := (others => '0'); --                tx_datak.tx_datak
		unused_tx_parallel_data : in  std_logic_vector(31 downto 0)  := (others => '0'); -- unused_tx_parallel_data.unused_tx_parallel_data
		rx_parallel_data        : out std_logic_vector(127 downto 0);                    --        rx_parallel_data.rx_parallel_data
		rx_datak                : out std_logic_vector(15 downto 0);                     --                rx_datak.rx_datak
		rx_errdetect            : out std_logic_vector(15 downto 0);                     --            rx_errdetect.rx_errdetect
		rx_disperr              : out std_logic_vector(15 downto 0);                     --              rx_disperr.rx_disperr
		rx_runningdisp          : out std_logic_vector(15 downto 0);                     --          rx_runningdisp.rx_runningdisp
		rx_patterndetect        : out std_logic_vector(15 downto 0);                     --        rx_patterndetect.rx_patterndetect
		rx_syncstatus           : out std_logic_vector(15 downto 0);                     --           rx_syncstatus.rx_syncstatus
		unused_rx_parallel_data : out std_logic_vector(31 downto 0)                      -- unused_rx_parallel_data.unused_rx_parallel_data
	);
end component;

component fastlink_small is
	port (
		pll_powerdown           : in  std_logic_vector(3 downto 0)   := (others => '0'); --           pll_powerdown.pll_powerdown
		tx_analogreset          : in  std_logic_vector(3 downto 0)   := (others => '0'); --          tx_analogreset.tx_analogreset
		tx_digitalreset         : in  std_logic_vector(3 downto 0)   := (others => '0'); --         tx_digitalreset.tx_digitalreset
		tx_pll_refclk           : in  std_logic_vector(0 downto 0)   := (others => '0'); --           tx_pll_refclk.tx_pll_refclk
		tx_serial_data          : out std_logic_vector(3 downto 0);                      --          tx_serial_data.tx_serial_data
		pll_locked              : out std_logic_vector(3 downto 0);                      --              pll_locked.pll_locked
		tx_std_coreclkin        : in  std_logic_vector(3 downto 0)   := (others => '0'); --        tx_std_coreclkin.tx_std_coreclkin
		tx_std_clkout           : out std_logic_vector(3 downto 0);                      --           tx_std_clkout.tx_std_clkout
		tx_cal_busy             : out std_logic_vector(3 downto 0);                      --             tx_cal_busy.tx_cal_busy
		reconfig_to_xcvr        : in  std_logic_vector(559 downto 0) := (others => '0'); --        reconfig_to_xcvr.reconfig_to_xcvr
		reconfig_from_xcvr      : out std_logic_vector(367 downto 0);                    --      reconfig_from_xcvr.reconfig_from_xcvr
		tx_parallel_data        : in  std_logic_vector(127 downto 0) := (others => '0'); --        tx_parallel_data.tx_parallel_data
		tx_datak                : in  std_logic_vector(15 downto 0)  := (others => '0'); --                tx_datak.tx_datak
		unused_tx_parallel_data : in  std_logic_vector(31 downto 0)  := (others => '0')  -- unused_tx_parallel_data.unused_tx_parallel_data
	);
end component;

component ip_alt_xcvr_reconfig is
	port (
		reconfig_busy             : out std_logic;                                          --      reconfig_busy.reconfig_busy
		mgmt_clk_clk              : in  std_logic                       := '0';             --       mgmt_clk_clk.clk
		mgmt_rst_reset            : in  std_logic                       := '0';             --     mgmt_rst_reset.reset
		reconfig_mgmt_address     : in  std_logic_vector(6 downto 0)    := (others => '0'); --      reconfig_mgmt.address
		reconfig_mgmt_read        : in  std_logic                       := '0';             --                   .read
		reconfig_mgmt_readdata    : out std_logic_vector(31 downto 0);                      --                   .readdata
		reconfig_mgmt_waitrequest : out std_logic;                                          --                   .waitrequest
		reconfig_mgmt_write       : in  std_logic                       := '0';             --                   .write
		reconfig_mgmt_writedata   : in  std_logic_vector(31 downto 0)   := (others => '0'); --                   .writedata
        --reconfig_to_xcvr          : out std_logic_vector(1119 downto 0);                    --   reconfig_to_xcvr.reconfig_to_xcvr
        --reconfig_from_xcvr        : in  std_logic_vector(735 downto 0)  := (others => '0')  -- reconfig_from_xcvr.reconfig_from_xcvr
        ch0_7_to_xcvr             : out std_logic_vector(559 downto 0);                    --   ch0_6_to_xcvr.reconfig_to_xcvr
        ch0_7_from_xcvr           : in  std_logic_vector(367 downto 0) := (others => '0'); -- ch0_6_from_xcvr.reconfig_from_xcvr
        ch8_15_to_xcvr             : out std_logic_vector(559 downto 0);                    --   ch7_9_to_xcvr.reconfig_to_xcvr
        ch8_15_from_xcvr           : in  std_logic_vector(367 downto 0) := (others => '0')  -- ch7_9_from_xcvr.reconfig_from_xcvr
);
end component;

component ip_altera_xcvr_reset_control is
	port (
		clock              : in  std_logic                    := '0';             --              clock.clk
		reset              : in  std_logic                    := '0';             --              reset.reset
		pll_powerdown      : out std_logic_vector(3 downto 0);                    --      pll_powerdown.pll_powerdown
		tx_analogreset     : out std_logic_vector(3 downto 0);                    --     tx_analogreset.tx_analogreset
		tx_digitalreset    : out std_logic_vector(3 downto 0);                    --    tx_digitalreset.tx_digitalreset
		tx_ready           : out std_logic_vector(3 downto 0);                    --           tx_ready.tx_ready
		pll_locked         : in  std_logic_vector(3 downto 0) := (others => '0'); --         pll_locked.pll_locked
		pll_select         : in  std_logic_vector(7 downto 0) := (others => '0'); --         pll_select.pll_select
		tx_cal_busy        : in  std_logic_vector(3 downto 0) := (others => '0'); --        tx_cal_busy.tx_cal_busy
		rx_analogreset     : out std_logic_vector(3 downto 0);                    --     rx_analogreset.rx_analogreset
		rx_digitalreset    : out std_logic_vector(3 downto 0);                    --    rx_digitalreset.rx_digitalreset
		rx_ready           : out std_logic_vector(3 downto 0);                    --           rx_ready.rx_ready
		rx_is_lockedtodata : in  std_logic_vector(3 downto 0) := (others => '0'); -- rx_is_lockedtodata.rx_is_lockedtodata
		rx_cal_busy        : in  std_logic_vector(3 downto 0) := (others => '0')  --        rx_cal_busy.rx_cal_busy
	);
end component;

component native_reset_tx is
	port (
		clock           : in  std_logic                    := '0';             --           clock.clk
		reset           : in  std_logic                    := '0';             --           reset.reset
		pll_powerdown   : out std_logic_vector(3 downto 0);                    --   pll_powerdown.pll_powerdown
		tx_analogreset  : out std_logic_vector(3 downto 0);                    --  tx_analogreset.tx_analogreset
		tx_digitalreset : out std_logic_vector(3 downto 0);                    -- tx_digitalreset.tx_digitalreset
		tx_ready        : out std_logic_vector(3 downto 0);                    --        tx_ready.tx_ready
		pll_locked      : in  std_logic_vector(3 downto 0) := (others => '0'); --      pll_locked.pll_locked
		pll_select      : in  std_logic_vector(7 downto 0) := (others => '0'); --      pll_select.pll_select
		tx_cal_busy     : in  std_logic_vector(3 downto 0) := (others => '0')  --     tx_cal_busy.tx_cal_busy
	);
end component;

end package transceiver_components;