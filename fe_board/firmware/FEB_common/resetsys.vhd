library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.daq_constants.all;

ENTITY resetsys is
PORT (
    clk_reset_rx_125 : in    std_logic; -- recovered clk from reset receiver
    clk_global_125   : in    std_logic; -- state transitions will be synchronised to this clock
    clk_156          : in    std_logic; 
    clk_free         : in    std_logic; -- independent, free running clock (not required for operation, used for phase measurement between clk_reset_rx and clk_global)
    state_out_156    : out   run_state_t; -- run state in sync to 156 clk
    state_out_125    : out   run_state_t; -- run state in sync to 125 clk
    reset_in         : in    std_logic; -- hard reset for testing, do not connect this to any "normal" reset
    resets_out       : out   std_logic_vector(15 downto 0); -- 16 bit reset mask, use this together with feb state .. example: nios_reset => (run_state=reset and resets(x)='1')  
    phase_out        : out   std_logic_vector(31 downto 0); -- phase between clk_reset_rx and clk_global
    data_in          : in    std_logic_vector(7 downto 0); -- 8b reset link input
    reset_bypass     : in    std_logic_vector(11 downto 0); -- bypass of reset link using nios & jtag (for setups without the genesis board) 
    run_number_out   : out   std_logic_vector(31 downto 0); -- run number from midas, updated on state run_prep
    fpga_id          : in    std_logic_vector(15 downto 0); -- input of fpga id, needed for addressed reset commands in setups with >1 FEBs
    terminated       : in    std_logic; -- changes run state from terminating to idle if set to 1  (data merger will set this if run was finished properly, signal will be synced to clk_reset_rx INSIDE this entity)
    testout          : out   std_logic_vector(5 downto 0)
);
END ENTITY;

architecture rtl of resetsys is

----------------signals---------------------

    -- states in sync with clk_reset_rx:
    -- (single std_logic for each state connected to state phase box,  phase box output is of type feb_run_state)
    signal ustate_idle_rx           : std_logic;
    signal ustate_run_prepare_rx    : std_logic;
    signal ustate_sync_rx           : std_logic;
    signal ustate_running_rx        : std_logic;
    signal ustate_terminating_rx    : std_logic;
    signal ustate_link_test_rx      : std_logic;
    signal ustate_sync_test_rx      : std_logic;
    signal ustate_reset_rx          : std_logic;
    signal ustate_out_of_DAQ_rx     : std_logic;

    -- terminated signal in sync to 125 clk of state controller
    signal terminated_125           : std_logic;

    signal state_controller_in      : std_logic_vector(7 downto 0);
    signal reset_bypass_125_rx      : std_logic_vector(11 downto 0);

----------------begin resetsys------------------------
BEGIN

    process(clk_reset_rx_125)
    begin
    if rising_edge(clk_reset_rx_125) then
        if ( reset_bypass_125_rx(8) = '1' ) then
            state_controller_in <= reset_bypass_125_rx(7 downto 0);
        else
            state_controller_in <= data_in;
        end if;
    end if;
    end process;

    -- sync terminated to 125 clk of state controller
    i_ff_sync : entity work.ff_sync
    generic map ( W => 1, N => 5 )
    PORT MAP (
        d(0)    => terminated,
        q(0)    => terminated_125,
        rst_n   => not reset_in,
        clk     => clk_reset_rx_125
    );

    i_state_controller : entity work.state_controller
    PORT MAP (
        clk                     => clk_reset_rx_125,
        reset                   => reset_in,
        reset_link_8bData       => state_controller_in,
        state_idle              => ustate_idle_rx,
        state_run_prepare       => ustate_run_prepare_rx,
        state_sync              => ustate_sync_rx,
        state_running           => ustate_running_rx,
        state_terminating       => ustate_terminating_rx,
        state_link_test         => ustate_link_test_rx,
        state_sync_test         => ustate_sync_test_rx,
        state_reset             => ustate_reset_rx,
        state_out_of_DAQ        => ustate_out_of_DAQ_rx,
        fpga_addr               => fpga_id,
        runnumber               => run_number_out,
        reset_mask              => resets_out,
        link_test_payload       => open,
        sync_test_payload       => open,
        terminated              => terminated_125
    );

    i_state_phase_box : entity work.state_phase_box
    PORT MAP (
        clk_global              => clk_global_125,
        clk_rx_reset            => clk_reset_rx_125,
        clk_free                => clk_free,
        reset                   => reset_in,
        phase                   => phase_out,
        -- states in sync to clk_rx_reset:
        state_idle_rx           => ustate_idle_rx,
        state_run_prepare_rx    => ustate_run_prepare_rx,
        state_sync_rx           => ustate_sync_rx,
        state_running_rx        => ustate_running_rx,
        state_terminating_rx    => ustate_terminating_rx,
        state_link_test_rx      => ustate_link_test_rx,
        state_sync_test_rx      => ustate_sync_test_rx,
        state_reset_rx          => ustate_reset_rx,
        state_out_of_DAQ_rx     => ustate_out_of_DAQ_rx,
        -- state in sync to clk_global:
        state_sync_global       => state_out_125
    );


    e_fifo_sync : entity work.fifo_sync
    generic map (
        DATA_WIDTH_g => run_state_t'length--,
    )
    port map (
        o_rdata     => state_out_156,
        i_rclk      => clk_156,
        i_reset_val => "0000000001",
        i_wdata     => '0' &
                       ustate_out_of_DAQ_rx &
                       ustate_reset_rx &
                       ustate_sync_test_rx &
                       ustate_link_test_rx &
                       ustate_terminating_rx &
                       ustate_running_rx &
                       ustate_sync_rx &
                       ustate_run_prepare_rx &
                       ustate_idle_rx,
        i_wclk      => clk_reset_rx_125,

        i_fifo_aclr => reset_in--,
    );

    e_fifo_sync2 : entity work.fifo_sync
    generic map (
        DATA_WIDTH_g => reset_bypass'length--,
    )
    port map (
        o_rdata     => reset_bypass_125_rx,
        i_rclk      => clk_reset_rx_125,
        i_wdata     => reset_bypass,
        i_wclk      => clk_156,
        i_fifo_aclr => reset_in--,
    );


    testout(0) <= ustate_idle_rx;
    testout(1) <= ustate_run_prepare_rx;
    testout(2) <= ustate_sync_rx;
    testout(3) <= ustate_running_rx;
    testout(4) <= ustate_terminating_rx;
    testout(5) <= ustate_reset_rx;

end architecture;
