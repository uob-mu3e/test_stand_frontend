----------------------------------------------------------------------------
-- Mupix SPI 
-- M. Mueller
-- JAN 2022

-- assembles correct sequence of 29 bit words for mp_ctrl_direct_spi.vhd
-- reads single bits out of the config mirrors
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;

use work.mupix.all;
use work.mudaq.all;


entity mp_ctrl_spi is
    generic( 
        DIRECT_SPI_FIFO_SIZE_g: positive := 4;
        N_CHIPS_PER_SPI_g: positive := 4--;
    );
    port(
        i_clk                   : in  std_logic;
        i_reset_n               : in  std_logic;

        o_spi_chip_selct_mask   : out std_logic_vector(N_CHIPS_PER_SPI_g-1 downto 0)--;
    );
end entity mp_ctrl_spi;

architecture RTL of mp_ctrl_spi is


begin

    process (i_clk, i_reset_n) is
    begin
        if(i_reset_n = '0') then

        elsif(rising_edge(i_clk)) then
         
        end if;
    end process;
end RTL;

-- the old thing: 


    -- signal mp_fifo_clear            : std_logic_vector(5 downto 0) := (others => '0');
    -- signal config_storage_input_data: std_logic_vector(32*5 + 31 downto 0);
    -- signal config_storage_in_all    : std_logic_vector(31 downto 0);
    -- signal config_storage_in_all_we : std_logic;
    -- signal is_writing               : std_logic_vector(5 downto 0);
    -- signal is_writing_this_round    : std_logic_vector(5 downto 0);
    -- signal config_storage_write     : std_logic_vector(5 downto 0);
    -- signal enable_shift_reg_6       : std_logic_vector(5 downto 0);
    -- signal clk1                     : std_logic_vector(5 downto 0);
    -- signal clk2                     : std_logic_vector(5 downto 0);
    -- signal clk_step                 : std_logic_vector(3 downto 0);

    -- signal rd_config                : std_logic;
    -- signal config_data              : std_logic_vector(5 downto 0);
    -- signal config_data29            : std_logic_vector(31 downto 0);
    -- signal config_data29_raw        : std_logic_vector(31 downto 0);
    -- signal waiting_for_load_round   : std_logic_vector(5 downto 0);

    -- signal slow_down                : std_logic_vector(15 downto 0);
    -- signal slow_down_buf            : std_logic_vector(31 downto 0);
    -- signal invert_29_bitpos         : std_logic;
    -- signal invert_csn               : std_logic;
    -- signal wait_cnt                 : std_logic_vector(15 downto 0);
    -- type mp_ctrl_state_type         is (idle, load_config, set_clks, writing, ld_spi_reg);
    -- signal mp_ctrl_state            : mp_ctrl_state_type;
    -- type spi_bit_state_type         is (beforepulse, duringpulse, afterpulse);
    -- signal spi_bit_state            : spi_bit_state_type;
    -- signal spi_busy                 : std_logic;

    -- signal spi_dout                 : std_logic;
    -- signal spi_clk                  : std_logic;
    -- signal spi_bitpos               : integer range 31 downto 0;
    -- signal chip_select_n            : std_logic_vector(11 downto 0);

    -- e_mupix_ctrl_config_storage : entity work.mupix_ctrl_config_storage
    -- port map(
    --     i_clk                       => i_clk,
    --     i_reset_n                   => i_reset_n,
    --     i_clr_fifo                  => mp_fifo_clear,

    --     i_data                      => config_storage_input_data,
    --     i_wrreq                     => config_storage_write,

    --     i_data_all                  => config_storage_in_all,
    --     i_wrreq_all                 => config_storage_in_all_we,

    --     o_data                      => config_data,
    --     o_is_writing                => is_writing,
    --     i_enable                    => enable_shift_reg_6,
    --     i_rdreq                     => rd_config--,
    -- );

    -- gen: for I in 0 to 31 generate
    --     config_data29(I) <= config_data29_raw(31-I) when invert_29_bitpos = '1' else config_data29_raw(I);
    -- end generate;

    -- process(i_clk, i_reset_n)
    -- begin
    --     if(i_reset_n = '0')then
    --         wait_cnt                <= (others => '0');
    --         mp_ctrl_state           <= idle;
    --         config_data29_raw       <= (others => '0');
    --         is_writing_this_round   <= (others => '0');
    --         waiting_for_load_round  <= (others => '0');
    --         clk1                    <= (others => '0');
    --         clk2                    <= (others => '0');
    --         clk_step                <= (others => '0');
    --         chip_select_n           <= (others => '1');
    --         spi_dout                <= '0';
    --         spi_clk                 <= '0';
    --         spi_bitpos              <= 0;
    --         -- o_csn                <= (others => '1'); -- do we want this ? (is a load signal)

    --     elsif(rising_edge(i_clk))then
    --         rd_config   <= '0';
    --         wait_cnt    <= wait_cnt + 1;
    --         o_mosi      <= (others => spi_dout);
    --         o_clock     <= (others => spi_clk);
    --         spi_busy    <= '1';
    --         --if(invert_csn = '1') then 
    --             o_csn       <= not chip_select_n;
    --         --else
    --         --    o_csn       <= chip_select_n;
    --         --end if;

    --         case mp_ctrl_state is
    --             when idle =>
    --                 waiting_for_load_round  <= (others => '0');
    --                 spi_busy                <= '0';

    --                 if(or_reduce(is_writing) = '1') then
    --                     mp_ctrl_state   <= load_config;
    --                     rd_config       <= '1';
    --                     wait_cnt        <= (others => '0');
    --                 end if;

    --             when load_config =>
    --                 chip_select_n                   <= (others => '1');
    --                 config_data29_raw               <= (others => '0');
    --                 clk_step                        <= (others => '0');
    --                 if(wait_cnt = x"0001") then
    --                     for I in 0 to 5 loop
    --                         config_data29_raw(I*3)  <= config_data(I); -- TODO: check order !!
    --                     end loop;
    --                     is_writing_this_round       <= is_writing;
    --                     mp_ctrl_state               <= set_clks;
    --                 end if;

    --             when set_clks =>
    --                 clk_step    <= clk_step + 1;
    --                 for I in 0 to 5 loop
    --                     if(is_writing_this_round(I)='1') then
    --                         waiting_for_load_round(I)   <= '1';
    --                         -- clocks (1.step 0 0, 2.step 1 0, 3.step 0 0, 4.step 0 1, 5.step 0 0)
    --                         case clk_step is
    --                             when x"0" => 
    --                                 config_data29_raw(I*3 + 1) <= '0';
    --                                 config_data29_raw(I*3 + 2) <= '0';
    --                             when x"1" => 
    --                                 config_data29_raw(I*3 + 1) <= '1';
    --                                 config_data29_raw(I*3 + 2) <= '0';
    --                             when x"2" => 
    --                                 config_data29_raw(I*3 + 1) <= '0';
    --                                 config_data29_raw(I*3 + 2) <= '0';
    --                             when x"3" => 
    --                                 config_data29_raw(I*3 + 1) <= '0';
    --                                 config_data29_raw(I*3 + 2) <= '1';
    --                             when x"4" => 
    --                                 config_data29_raw(I*3 + 1) <= '0';
    --                                 config_data29_raw(I*3 + 2) <= '0';
    --                             when others =>
                                    
    --                         end case;
    --                     else
    --                         config_data29_raw(I*3 + 1) <= '0';
    --                         config_data29_raw(I*3 + 2) <= '0';
    --                     end if;
    --                 end loop;

    --                 if(or_reduce(is_writing)='1' and clk_step=x"5") then -- next bit
    --                     mp_ctrl_state <= load_config;
    --                     rd_config     <= '1';
    --                     wait_cnt      <= (others => '0');
    --                 end if;

    --                 if(or_reduce(is_writing)='0') then -- done
    --                     mp_ctrl_state <= idle;
    --                 end if;
 
    --                 if(or_reduce(is_writing)='0' and clk_step=x"5") then -- extra round for load conf
    --                     mp_ctrl_state <= writing;
    --                     spi_bit_state <= beforepulse;
    --                     wait_cnt      <= (others => '0');
    --                     config_data29_raw <= (19 => '1', others => '0');
    --                 end if;

    --                 if(clk_step=x"6") then -- extra round for load
    --                     mp_ctrl_state <= writing;
    --                     spi_bit_state <= beforepulse;
    --                     wait_cnt      <= (others => '0');
    --                     config_data29_raw <= (19 => '1', others => '0');
    --                     for I in 0 to 5 loop
    --                          -- set ld bits that have written something since last idle
    --                         config_data29_raw(18 + I) <= waiting_for_load_round(I);
    --                     end loop;
    --                     config_data29_raw(19) <= '1';
    --                 end if;

    --                 if(clk_step=x"7") then -- extra round for load remove
    --                     mp_ctrl_state <= writing;
    --                     spi_bit_state <= beforepulse;
    --                     wait_cnt      <= (others => '0');
    --                     config_data29_raw <= (19 => '1', others => '0');
    --                 end if;

    --                 if(clk_step=x"0" or clk_step=x"1" or clk_step=x"2" or clk_step=x"3" or clk_step=x"4") then -- writing
    --                     mp_ctrl_state <= writing;
    --                     spi_bit_state <= beforepulse;
    --                     wait_cnt      <= (others => '0');
    --                 end if;

    --             when writing =>
    --                 spi_dout                <= config_data29(spi_bitpos);
    --                 case spi_bit_state is
    --                     when beforepulse => 
    --                         spi_clk <= '0';
    --                         if(wait_cnt(14 downto 0) = slow_down(15 downto 1)) then -- wait_cnt = slow_down/2
    --                             wait_cnt        <= (others => '0');
    --                             spi_bit_state   <= duringpulse;
    --                         end if;
    --                     when duringpulse =>
    --                         spi_clk <= '1';
    --                         if(wait_cnt = slow_down) then
    --                             wait_cnt        <= (others => '0');
    --                             spi_bit_state   <= afterpulse;
    --                         end if;
    --                     when afterpulse =>
    --                         spi_clk <= '0';
    --                         if(wait_cnt(14 downto 0) = slow_down(15 downto 1)) then -- wait_cnt = slow_down/2
    --                             wait_cnt            <= (others => '0');
    --                             spi_bit_state       <= beforepulse;
    --                             if(spi_bitpos=31) then
    --                                 mp_ctrl_state   <= ld_spi_reg;
    --                                 spi_bitpos      <= 0;
    --                             else
    --                                 spi_bitpos      <= spi_bitpos + 1;
    --                             end if;
    --                         end if;
    --                     when others =>
    --                         spi_bit_state <= beforepulse;
    --                 end case;

    --             when ld_spi_reg =>
    --                 if(wait_cnt=slow_down) then
    --                     chip_select_n   <= chip_select_mask;
    --                 end if;
    --                 if(wait_cnt = slow_down + slow_down) then 
    --                     mp_ctrl_state   <= set_clks;
    --                     chip_select_n   <= (others => '1');
    --                 end if;
    --             when others =>
    --                 mp_ctrl_state <= idle;
    --         end case;
            
    --     end if;
    -- end process;