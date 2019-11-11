library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.daq_constants.all;

ENTITY resetsys is
PORT (
    i_data_125_rx       : in    std_logic_vector(7 downto 0);
    i_reset_125_rx_n    : in    std_logic;
    i_clk_125_rx        : in    std_logic;

    o_state_125         : out   run_state_t;
    i_reset_125_n       : in    std_logic;
    i_clk_125           : in    std_logic;

    o_state_156         : out   run_state_t;
    i_reset_156_n       : in    std_logic;
    i_clk_156           : in    std_logic;

    resets_out          : out   std_logic_vector(15 downto 0); -- 16 bit reset mask, use this together with feb state .. example: nios_reset => (run_state=reset and resets(x)='1')
    reset_bypass        : in    std_logic_vector(11 downto 0); -- bypass of reset link using nios & jtag (for setups without the genesis board)
    run_number_out      : out   std_logic_vector(31 downto 0); -- run number from midas, updated on state run_prep
    fpga_id             : in    std_logic_vector(15 downto 0); -- input of fpga id, needed for addressed reset commands in setups with >1 FEBs
    terminated          : in    std_logic; -- changes run state from terminating to idle if set to 1  (data merger will set this if run was finished properly, signal will be synced to clk_reset_rx INSIDE this entity)
    testout             : out   std_logic_vector(5 downto 0);

    o_phase             : out   std_logic_vector(31 downto 0);
    i_reset_n           : in    std_logic;
    i_clk               : in    std_logic--;
);
END ENTITY;

architecture rtl of resetsys is

    signal state_125_rx : run_state_t;

    -- terminated signal in sync to clk_125_rx of state controller
    signal terminated_125_rx : std_logic;

    signal state_controller_in : std_logic_vector(7 downto 0);
    signal reset_bypass_125_rx : std_logic_vector(11 downto 0);

----------------begin resetsys------------------------
BEGIN

    process(i_clk_125_rx)
    begin
    if rising_edge(i_clk_125_rx) then
        if ( reset_bypass_125_rx(8) = '1' ) then
            state_controller_in <= reset_bypass_125_rx(7 downto 0);
        else
            state_controller_in <= i_data_125_rx;
        end if;
    end if;
    end process;

    -- sync terminated to 125 clk of state controller
    i_ff_sync : entity work.ff_sync
    generic map ( W => 1, N => 5 )
    PORT MAP (
        d(0)    => terminated,
        q(0)    => terminated_125_rx,
        rst_n   => i_reset_125_rx_n,
        clk     => i_clk_125_rx--,
    );

    -- decode state from rx
    i_state_controller : entity work.state_controller
    PORT MAP (
        reset_link_8bData       => state_controller_in,
        fpga_addr               => fpga_id,
        runnumber               => run_number_out,
        reset_mask              => resets_out,
        link_test_payload       => open,
        sync_test_payload       => open,
        terminated              => terminated_125_rx,

        o_state                 => state_125_rx,

        i_reset_n               => i_reset_125_rx_n,
        i_clk                   => i_clk_125_rx--,
    );

    -- measure phase between clk_125_rx and clk_125
    -- sync state from clk_125_rx to clk_125
    i_state_phase_box : entity work.state_phase_box
    PORT MAP (
        i_state_125_rx      => state_125_rx,
        i_clk_125_rx        => i_clk_125_rx,

        o_state_125         => o_state_125,
        i_reset_125_n       => i_reset_125_n,
        i_clk_125           => i_clk_125,

        phase               => o_phase,
        i_reset_n           => i_reset_n,
        i_clk               => i_clk--,
    );

    -- sync state from clk_125_rx to clk_156
    e_fifo_sync : entity work.fifo_sync
    generic map (
        DATA_WIDTH_g => run_state_t'length--,
    )
    port map (
        o_rdata     => o_state_156,
        i_rclk      => i_clk_156,
        i_reset_val => RUN_STATE_IDLE,

        i_wdata     => state_125_rx,
        i_wclk      => i_clk_125_rx,

        i_fifo_aclr => not i_reset_156_n--,
    );

    e_fifo_sync2 : entity work.fifo_sync
    generic map (
        DATA_WIDTH_g => reset_bypass'length--,
    )
    port map (
        o_rdata     => reset_bypass_125_rx,
        i_rclk      => i_clk_125_rx,

        i_wdata     => reset_bypass,
        i_wclk      => i_clk_156,

        i_fifo_aclr => not i_reset_125_rx_n--,
    );

    testout <= state_125_rx(testout'range);

end architecture;
