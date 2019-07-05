library ieee;
use ieee.std_logic_1164.all;

package protocol is

    type data_merger_state is (idle, sending_data, sending_slowcontrol);
    --type feb_state is (idle, run_prep, sync, running, terminating, link_test, sync_test, reset_state, out_of_DAQ);

    constant HEADER_K:    std_logic_vector(31 downto 0) := x"bcbcbcbc";
    constant DATA_HEADER_ID:    std_logic_vector(5 downto 0) := "111010";
    constant DATA_SUB_HEADER_ID:    std_logic_vector(5 downto 0) := "111111";
    constant ACTIVE_SIGNAL_HEADER_ID:    std_logic_vector(5 downto 0) := "111101";
    constant RUN_TAIL_HEADER_ID:    std_logic_vector(5 downto 0) := "111110";
    constant TIMING_MEAS_HEADER_ID:    std_logic_vector(5 downto 0) := "111100";
    constant SC_HEADER_ID:    std_logic_vector(5 downto 0) := "111011";

end package protocol;