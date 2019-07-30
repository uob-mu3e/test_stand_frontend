library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

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

signal rdreq : std_logic;
signal wrreq : std_logic;
signal empty : std_logic;
signal fifo_data_out : run_state_t;

begin
    sync_fifo : entity work.ip_dcfifo
    generic map (
        ADDR_WIDTH => 4,
        DATA_WIDTH => 10--,
    )
    port map (
        data            => state_rx,
        rdclk           => clk_156,
        rdreq           => rdreq,
        wrclk           => clk_rx,
        wrreq           => wrreq,
        q               => fifo_data_out,
        rdempty         => empty,
        rdusedw         => open,
        wrfull          => open,
        wrusedw         => open
    );

process (clk_156, reset_n)
    begin
    if rising_edge(clk_156) then
        if (reset_n = '0') then 
            rdreq <= '0';
            state_156 <= RUN_STATE_RESET; -- RUN_STATE_IDLE ??
        elsif(empty = '0') then 
            rdreq <= '1';
            state_156 <= fifo_data_out;
        else
            rdreq <= '0';
        end if;
    end if;
end process;

wrreq <= '1';

END architecture;
