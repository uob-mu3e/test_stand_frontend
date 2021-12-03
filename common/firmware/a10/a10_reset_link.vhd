--
-- author : Marius Koeppel
-- date : 2021-11
--

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mudaq.all;
use work.a10_pcie_registers.all;


entity a10_reset_link is
generic (
	g_XCVR2_CHANNELS    : integer := 0--;
);
port (
    o_xcvr_tx_data      : out   std_logic_vector(g_XCVR2_CHANNELS*8-1 downto 0) := (others => '0');
    o_xcvr_tx_datak     : out   std_logic_vector(3 downto 0);
    
    i_reset_run_number  : in    std_logic_vector(31 downto 0);
    i_reset_ctl         : in    std_logic_vector(31 downto 0);
    i_clk               : in    std_logic;
	
	o_state_out			: out   std_logic_vector(31 downto 0);

    i_reset_n           : in    std_logic--;
);
end entity;

architecture arch of a10_reset_link is

	type reset_state_type is (idle, sync, start_run, rNB0, rNB1, rNB2, rNB3, end_run);
    signal reset_state : reset_state_type;
	signal reg_run_prepare, reg_sync, reg_start_run, reg_end_run,reg_abort_run : std_logic;
	signal cur_output : std_logic_vector(2 downto 0);
	signal xcvr_tx_data : std_logic_vector(7 downto 0);

begin

	o_xcvr_tx_data(7 downto 0)   <= xcvr_tx_data when cur_output = "000" else 
									xcvr_tx_data when cur_output = "111" 
									else x"BC";
	o_xcvr_tx_data(15 downto 8)  <= xcvr_tx_data when cur_output = "001" else
									xcvr_tx_data when cur_output = "111" 
									else x"BC";
	o_xcvr_tx_data(23 downto 16) <= xcvr_tx_data when cur_output = "010" else 
									xcvr_tx_data when cur_output = "111" 
									else x"BC";
	o_xcvr_tx_data(31 downto 24) <= xcvr_tx_data when cur_output = "011" else 
									xcvr_tx_data when cur_output = "111" 
									else x"BC";
							
	o_xcvr_tx_datak(0)  <= 	'0' when cur_output = "000" else 
							'0' when cur_output = "111" 
							else '1';
	o_xcvr_tx_datak(1)  <= 	'0' when cur_output = "001" else
							'0' when cur_output = "111" 
							else '1';
	o_xcvr_tx_datak(2)  <= 	'0' when cur_output = "010" else 
							'0' when cur_output = "111" 
							else '1';
	o_xcvr_tx_datak(3)  <= 	'0' when cur_output = "011" else 
							'0' when cur_output = "111" 
							else '1';
		
	-- reset link process
    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        xcvr_tx_data 	<= x"BC";
		reset_state 	<= idle;
		cur_output 		<= "111";
		o_state_out		<= x"00000000";
        --
    elsif rising_edge(i_clk) then
		xcvr_tx_data	<= x"BC";
		
		cur_output 		<= i_reset_ctl(RESET_LINK_FEB_RANGE);
		
		reg_run_prepare <= i_reset_ctl(RESET_LINK_RUN_PREPARE_BIT);
		reg_sync 		<= i_reset_ctl(RESET_LINK_SYNC_BIT);
		reg_start_run	<= i_reset_ctl(RESET_START_RUN_BIT);
		reg_end_run 	<= i_reset_ctl(RESET_END_RUN_BIT);
		reg_abort_run 	<= i_reset_ctl(RESET_LINK_ABORT_RUN_BIT);
        
		-- abortding should be always possible
		if ( reg_abort_run = '0' and i_reset_ctl(RESET_LINK_ABORT_RUN_BIT) = '1' ) then
			xcvr_tx_data <= RESET_LINK_ABORT_RUN;
			o_state_out  <= x"00000000";
			reset_state  <= idle;
		else
			case reset_state is
					
					when idle =>
						-- wait for run prepare
						if ( reg_run_prepare = '0' and i_reset_ctl(RESET_LINK_RUN_PREPARE_BIT) = '1' ) then
							xcvr_tx_data <= RESET_LINK_RUN_PREPARE;
							reset_state  <= rNB0;
							o_state_out  <= x"00000001";
						end if;
					
					when rNB0 =>
						xcvr_tx_data <= i_reset_run_number(7 downto 0);
						reset_state  <= rNB1;
						
					when rNB1 =>
						xcvr_tx_data <= i_reset_run_number(15 downto 8);
						reset_state  <= rNB2;
						
					when rNB2 =>
						xcvr_tx_data <= i_reset_run_number(23 downto 16);
						reset_state  <= rNB3;
						
					when rNB3 =>
						xcvr_tx_data <= i_reset_run_number(31 downto 24);
						reset_state  <= sync;
						o_state_out  <= x"00000002";
						
					when sync =>
						-- wait for sync
						if ( reg_sync = '0' and i_reset_ctl(RESET_LINK_SYNC_BIT) = '1' ) then
							xcvr_tx_data <= RESET_LINK_SYNC;
							reset_state  <= start_run;
							o_state_out  <= x"00000003";
						end if;
					
					when start_run =>
						-- wait for start run
						if ( reg_start_run = '0' and i_reset_ctl(RESET_START_RUN_BIT) = '1' ) then
							xcvr_tx_data <= RESET_LINK_START_RUN;
							reset_state  <= end_run;
							o_state_out  <= x"00000004";
						end if;
						
					when end_run =>
						-- wait for end run
						if ( reg_end_run = '0' and i_reset_ctl(RESET_END_RUN_BIT) = '1' ) then
							xcvr_tx_data <= RESET_LINK_END_RUN;
							reset_state  <= idle;
							o_state_out  <= x"00000000";
						end if;
						
					when others =>
						reset_state  <= idle;
				end case;
		end if;
	end if;		
	end process;

end architecture;
