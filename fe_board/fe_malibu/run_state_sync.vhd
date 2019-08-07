library ieee;
use ieee.std_logic_1164.all;

use work.daq_constants.all;

ENTITY sync_reset is
PORT (
    clk_rx                  : in    std_logic;
    clk_156                 : in    std_logic;
    reset_n                 : in    std_logic;
    -- states in sync to clk_rx_reset:
    state_rx                : in    run_state_t;
    -- states in sync to clk_global:
    state_156               : out   run_state_t
);
END ENTITY;

architecture rtl of sync_reset is

    signal fifo_empty : std_logic;
    signal fifo_q : run_state_t;

begin

    e_fifo : entity work.ip_dcfifo
    generic map (
        ADDR_WIDTH => 3,
        DATA_WIDTH => run_state_t'length--,
    )
    port map (
        aclr        => not reset_n,
        data        => state_rx,
        rdclk       => clk_156,
        rdreq       => not fifo_empty,
        wrclk       => clk_rx,
        wrreq       => '1',
        q           => fifo_q,
        rdempty     => fifo_empty,
        rdusedw     => open,
        wrfull      => open,
        wrusedw     => open
    );

    process(clk_156, reset_n)
    begin
    if ( reset_n = '0' ) then
        state_156 <= RUN_STATE_RESET; -- RUN_STATE_IDLE ??
        --
    elsif rising_edge(clk_156) then
        if ( fifo_empty = '0' ) then
            state_156 <= fifo_q;
        end if;
        --
    end if;
    end process;

END architecture;
