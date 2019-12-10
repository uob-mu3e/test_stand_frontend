-- receive run control signals from FEBs
-- TODO more than one FEB


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.daq_constants.all;

ENTITY run_control is
    generic (
        N_LINKS_g : positive := 4--;
    );
    PORT(
        i_clk:                              in  std_logic; -- receive clock (156.25 MHz)
        i_reset_n:                          in  std_logic;
        i_aligned:                          in  std_logic_vector(31 downto 0); -- word alignment achieved
        i_data:                             in  std_logic_vector(32*N_LINKS_g-1 downto 0); -- optical from frontend board
        i_datak:                            in  std_logic_vector(4*N_LINKS_g-1 downto 0);
        i_link_enable:                      in  std_logic_vector(31 downto 0);
        i_addr:                             in  std_logic_vector(31 downto 0);
        i_run_number:                       in  std_logic_vector(23 downto 0);
        o_run_number:                       out std_logic_vector(31 downto 0);
        o_runNr_ack:                        out std_logic_vector(31 downto 0)--;
);
END ENTITY run_control;

architecture rtl of run_control is

signal FEB_status :                             std_logic_vector(32*N_LINKS_g-1 downto 0);

BEGIN
    g_link_listener : for i in N_LINKS_g-1 downto 0 generate
        begin
            e_link_listener : entity work.run_control_link_listener
            port map (
                i_clk           => i_clk, 
                i_reset_n       => i_reset_n,
                i_aligned       => i_aligned(i),
                i_data          => i_data(31+32*i downto 32*i),
                i_datak         => i_datak(3+4*i downto 4*i),
                o_FEB_status    => FEB_status(31+32*i downto 32*i)--,
            );
    end generate;

    process (i_clk, i_reset_n)

    begin
        if (i_reset_n = '0') then 
            o_run_number                    <= (others => '0');
            o_runNr_ack                     <= (others => '0');
            
        elsif (rising_edge(i_clk)) then
                for J in 0 to N_LINKS_g-1 loop
                    if (FEB_status(23+J*32 downto 0+J*32) = i_run_number and i_link_enable(J)='1') then
                        o_runNr_ack(J)      <= '1';
                    else
                        o_runNr_ack(J)      <= '0';
                    end if;
                end loop;
                
                o_run_number                <= x"00" & FEB_status(23+to_integer(unsigned(i_addr))*32 downto to_integer(unsigned(i_addr))*32);
        end if;
    end process;

END rtl;