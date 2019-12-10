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
        i_aligned:                          in  std_logic_vector(N_LINKS_g-1 downto 0); -- word alignment achieved
        i_data:                             in  std_logic_vector(32*N_LINKS_g-1 downto 0); -- optical from frontend board
        i_datak:                            in  std_logic_vector(4*N_LINKS_g-1 downto 0);
        i_link_mask:                        in  std_logic_vector(31 downto 0);
        i_addr:                             in  std_logic_vector(31 downto 0);
        o_run_number:                       out std_logic_vector(31 downto 0);
        o_link_active:                      out std_logic_vector(31 downto 0);
        o_runNr_ack:                        out std_logic_vector(31 downto 0)--;
);
END ENTITY run_control;

architecture rtl of run_control is

signal run_prep_acknowledge_received :          std_logic_vector(31 downto 0);
signal end_of_run_received :                    std_logic_vector(31 downto 0);
signal run_number :                             std_logic_vector(24*N_LINKS_g-1 downto 0);
signal o_FEB_status :                           std_logic_vector(32*N_LINKS_g-1 downto 0);

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
                o_FEB_status    => o_FEB_status(31+32*i downto 32*i)--,
            );
    end generate;
END rtl;