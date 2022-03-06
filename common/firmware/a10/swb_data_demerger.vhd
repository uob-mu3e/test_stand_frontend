-- demerging data and slowcontrol from FEB on the switching board
-- Martin Mueller, May 2019

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity swb_data_demerger is
port (
    i_aligned:                  in  std_logic; -- word alignment achieved
    i_data                      : in    work.mu3e.link_t; -- optical from frontend board
    i_fifo_almost_full:         in  std_logic;

    o_data:                     out std_logic_vector(31 downto 0); -- to sorting fifos
    o_datak:                    out std_logic_vector(3 downto 0); -- to sorting fifos
    o_sc:                       out std_logic_vector(31 downto 0); -- slowcontrol from frontend board
    o_sck:                      out std_logic_vector(3 downto 0); -- slowcontrol from frontend board
    o_rc:                       out std_logic_vector(31 downto 0);
    o_rck:                      out std_logic_vector(3 downto 0);
    o_fpga_id:                  out std_logic_vector(15 downto 0);  -- FPGA ID of the connected frontend board

    i_reset:                    in  std_logic;
    i_clk:                      in  std_logic--;
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
        o_data              <= x"000000"& work.util.K28_5;
        o_datak             <= "0001";
        o_sc                <= x"000000"& work.util.K28_5;
        o_sck               <= "0001";
        o_rc                <= x"000000"& work.util.K28_5;
        o_rck               <= "0001";

    elsif rising_edge(i_clk) then
        o_data              <= x"000000"& work.util.K28_5;
        o_datak             <= "0001";
        o_sc                <= x"000000"& work.util.K28_5;
        o_sck               <= "0001";
        o_rc                <= x"000000"& work.util.K28_5;
        o_rck               <= "0001";

        case demerge_state is

        when idle =>
            if ( i_data.datak = "0001" and i_data.data(7 downto 0) /= work.util.K28_5 and i_data.data(7 downto 0) /= work.util.K28_4 ) then
                o_rc <= i_data.data;
                o_rck <= i_data.datak;
            elsif ( i_data.datak(3 downto 0) = "0001" and i_data.data(7 downto 0) = work.util.K28_5 and i_data.data(31 downto 29) = "111" and i_fifo_almost_full='0' ) then -- Mupix or MuTrig preamble
                o_fpga_id <= i_data.data(23 downto 8);
                demerge_state           <= receiving_data;
                o_data <= i_data.data;
                o_datak <= i_data.datak;
            elsif ( i_data.datak(3 downto 0) = "0001" and i_data.data(7 downto 0) = work.util.K28_5 and i_data.data(31 downto 26) = "000111" ) then -- SC preamble
                o_fpga_id                 <= i_data.data(23 downto 8);
                demerge_state           <= receiving_slowcontrol;
                slowcontrol_type <= i_data.data(25 downto 24);
                o_sc <= i_data.data;
                o_sck <= i_data.datak;
            end if;

        when receiving_data =>
            if ( i_data.datak = "0001" and i_data.data(7 downto 0) /= work.util.K28_5 and i_data.data(7 downto 0) /= work.util.K28_4 ) then
                o_rc <= i_data.data;
                o_rck <= i_data.datak;
            elsif ( i_data.data(7 downto 0) = work.util.K28_4 and i_data.datak = "0001" ) then
                demerge_state       <= idle;
                o_data <= i_data.data;
                o_datak <= i_data.datak;
            else
                 o_data <= i_data.data;
                 o_datak <= i_data.datak;
            end if;

        when receiving_slowcontrol =>
            if ( i_data.datak = "0001" and i_data.data(7 downto 0) /= work.util.K28_5 and i_data.data(7 downto 0) /= work.util.K28_4 ) then
                o_rc <= i_data.data;
                o_rck <= i_data.datak;
            elsif ( i_data.data(7 downto 0) = work.util.K28_4 and i_data.datak = "0001" ) then
                demerge_state       <= idle;
                o_sc <= i_data.data;
                o_sck <= i_data.datak;
            else
                o_sc <= i_data.data;
                o_sck <= i_data.datak;
            end if;

        when others =>
            demerge_state <= idle;

        end case;

    end if;
    end process;

end architecture;
