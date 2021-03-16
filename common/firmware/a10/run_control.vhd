-- receive run control signals from FEBs
-- TODO more than one FEB


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.daq_constants.all;
use ieee.std_logic_misc.all;


entity run_control is
generic (
        N_LINKS_g : integer := 4--;
);
port (
        i_clk:                              in  std_logic; -- receive clock (156.25 MHz)
        i_reset_ack_seen_n:                 in  std_logic;
        i_reset_run_end_n:                  in  std_logic;
        i_buffers_empty:                    in  std_logic_vector(31 downto 0);
        i_aligned:                          in  std_logic_vector(31 downto 0); -- word alignment achieved
        i_data:                             in  work.util.slv32_array_t(N_LINKS_g-1 downto 0); -- optical from frontend board
        i_datak:                            in  work.util.slv4_array_t(N_LINKS_g-1 downto 0);
        i_link_enable:                      in  std_logic_vector(31 downto 0);
        i_addr:                             in  std_logic_vector(31 downto 0);
        i_run_number:                       in  std_logic_vector(23 downto 0);
        o_run_number:                       out std_logic_vector(31 downto 0);
        o_runNr_ack:                        out std_logic_vector(31 downto 0);
        o_run_stop_ack:                     out std_logic_vector(31 downto 0);
        o_buffers_empty:                    out std_logic_vector(31 downto 0);
        o_feb_merger_timeout:               out std_logic_vector(31 downto 0)--;
);
end entity;

architecture rtl of run_control is

signal FEB_status :                             std_logic_vector(26*N_LINKS_g-1 downto 0);
signal feb_merger_timeouts :                    std_logic_vector(N_LINKS_g-1 downto 0);
signal CNT_feb_merger_timeouts :                unsigned(31 downto 0);

BEGIN
   o_feb_merger_timeout <= std_logic_vector(CNT_feb_merger_timeouts);

    g_link_listener : for i in N_LINKS_g-1 downto 0 generate
        begin
            e_link_listener : entity work.run_control_link_listener
            port map (
                i_clk                   => i_clk, 
                i_reset_ack_seen_n      => i_reset_ack_seen_n,
                i_reset_run_end_n       => i_reset_run_end_n,
                i_aligned               => i_aligned(i),
                i_data                  => i_data(i),
                i_datak                 => i_datak(i),
                o_merger_timeout        => feb_merger_timeouts(i),
                o_FEB_status            => FEB_status(25+26*i downto 26*i)--,
            );
    end generate;
    
    feb_merger_timeout : process(i_clk, i_reset_ack_seen_n, i_reset_run_end_n)
    begin
    if ( i_reset_ack_seen_n = '0' ) then
        CNT_feb_merger_timeouts  <= (others => '0');
    elsif rising_edge(i_clk) then
      if(or_reduce(feb_merger_timeouts)='1') then
         CNT_feb_merger_timeouts <= CNT_feb_merger_timeouts + 1;
      end if;
    end if;
    end process;

    process (i_clk)
    begin
        if (rising_edge(i_clk)) then
            if (i_reset_ack_seen_n = '0') then 
                o_run_number                    <= (others => '0');
                o_runNr_ack                     <= (others => '0');
            end if;
            
            if (i_reset_run_end_n = '0') then
                o_run_stop_ack                  <= (others => '0');
            end if;
            
                if (and_reduce(i_buffers_empty) = '1') then
                    o_buffers_empty         <= (0 => '1', others => '0');
                else 
                    o_buffers_empty         <= (others => '0');
                end if;
                
                for J in 0 to N_LINKS_g-1 loop
                    if (FEB_status(25+26*J)='1' and (FEB_status(23+J*26 downto 0+J*26) = i_run_number)) then
                        o_runNr_ack(J)      <= '1';
                    else
                        o_runNr_ack(J)      <= '0';
                    end if;
                    
                    o_run_stop_ack(J)       <= FEB_status(24+J*26);
                end loop;
                
                o_run_number                <= "000000" & FEB_status(25+to_integer(unsigned(i_addr))*26 downto to_integer(unsigned(i_addr))*26);
        end if;
        
        
    end process;

end architecture;
