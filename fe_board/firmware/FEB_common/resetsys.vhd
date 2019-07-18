library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.daq_constants.all;


ENTITY resetsys is 
    PORT(
        clk:                    in  std_logic; -- 125 Mhz clock from reset transceiver
        reset_in :              in  std_logic; -- hard reset for testing
		  resets_out :				  out std_logic_vector(15 downto 0);
        data_in :    		     in  std_logic_vector(7 downto 0); -- input reset line
		  state_out :				  out feb_run_state;
		  run_number_out :   	  out std_logic_vector(31 downto 0);
		  testout : 				  out std_logic
    );
END ENTITY resetsys;

architecture rtl of resetsys is


----------------signals---------------------
    signal run_state : feb_run_state;
    
----------------begin resetsys------------------------
BEGIN

i_state_controller: entity work.state_controller
    PORT MAP(
        clk                    => clk,
        reset                  => reset_in,
        reset_link_8bData      => data_in,
        state_idle             => testout,
        state_run_prepare      => open,
        state_sync             => open,
        state_running          => open,
        state_terminating      => open,
        state_link_test        => open,
        state_sync_test        => open,
        state_reset            => open,
        state_out_of_DAQ       => open,
        fpga_addr              => x"CAFE",
        runnumber              => run_number_out,
        reset_mask             => resets_out,
        link_test_payload      => open,
        sync_test_payload      => open,
        terminated             => '0'
    );

END rtl;
