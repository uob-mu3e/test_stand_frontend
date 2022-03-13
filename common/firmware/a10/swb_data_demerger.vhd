-- demerging data and slowcontrol from FEB on the switching board
-- Martin Mueller, May 2019

library ieee;
use ieee.std_logic_1164.all;

entity swb_data_demerger is
port (
    i_aligned                   : in    std_logic; -- word alignment achieved
    i_data                      : in    work.mu3e.link_t; -- optical from frontend board
    i_fifo_almost_full          : in    std_logic;

    o_data                      : out   work.mu3e.link_t; -- to sorting fifos
    o_sc                        : out   work.mu3e.link_t; -- slowcontrol from frontend board
    o_rc                        : out   work.mu3e.link_t;
    o_fpga_id                   : out   std_logic_vector(15 downto 0);  -- FPGA ID of the connected frontend board

    i_reset                     : in    std_logic;
    i_clk                       : in    std_logic--;
);
end entity;

architecture arch of swb_data_demerger is

----------------signals---------------------
    type   data_demerge_state is (idle, receiving_data, receiving_slowcontrol);
    signal demerge_state :          data_demerge_state;
    signal slowcontrol_type :       std_logic_vector(1 downto 0);

----------------begin data_demerge------------------------
begin

    process(i_clk, i_reset, i_aligned)
    begin
    if ( i_reset = '1' or i_aligned = '0' ) then
        demerge_state       <= idle;
        o_data <= work.mu3e.LINK_IDLE;
        o_sc <= work.mu3e.LINK_IDLE;
        o_rc <= work.mu3e.LINK_IDLE;

    elsif rising_edge(i_clk) then
        o_data <= work.mu3e.LINK_IDLE;
        o_sc <= work.mu3e.LINK_IDLE;
        o_rc <= work.mu3e.LINK_IDLE;

        case demerge_state is

        when idle =>
            if ( i_data.datak = "0001" and i_data.data(7 downto 0) /= work.util.K28_5 and i_data.data(7 downto 0) /= work.util.K28_4 ) then
                o_rc <= i_data;
            elsif ( i_data.datak(3 downto 0) = "0001" and i_data.data(7 downto 0) = work.util.K28_5 and i_data.data(31 downto 29) = "111" and i_fifo_almost_full='0' ) then -- Mupix or MuTrig preamble
                o_fpga_id <= i_data.data(23 downto 8);
                demerge_state           <= receiving_data;
                o_data <= i_data;
            elsif ( i_data.datak(3 downto 0) = "0001" and i_data.data(7 downto 0) = work.util.K28_5 and i_data.data(31 downto 26) = "000111" ) then -- SC preamble
                o_fpga_id                 <= i_data.data(23 downto 8);
                demerge_state           <= receiving_slowcontrol;
                slowcontrol_type <= i_data.data(25 downto 24);
                o_sc <= i_data;
            end if;

        when receiving_data =>
            if ( i_data.datak = "0001" and i_data.data(7 downto 0) /= work.util.K28_5 and i_data.data(7 downto 0) /= work.util.K28_4 ) then
                o_rc <= i_data;
            elsif ( i_data.data(7 downto 0) = work.util.K28_4 and i_data.datak = "0001" ) then
                demerge_state       <= idle;
                o_data <= i_data;
            else
                 o_data <= i_data;
            end if;

        when receiving_slowcontrol =>
            if ( i_data.datak = "0001" and i_data.data(7 downto 0) /= work.util.K28_5 and i_data.data(7 downto 0) /= work.util.K28_4 ) then
                o_rc <= i_data;
            elsif ( i_data.data(7 downto 0) = work.util.K28_4 and i_data.datak = "0001" ) then
                demerge_state       <= idle;
                o_sc <= i_data;
            else
                o_sc <= i_data;
            end if;

        when others =>
            demerge_state <= idle;

        end case;

    end if;
    end process;

end architecture;
