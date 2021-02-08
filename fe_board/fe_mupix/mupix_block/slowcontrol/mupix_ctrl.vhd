----------------------------------------------------------------------------
-- Mupix control
-- M. Mueller
-- JAN 2021
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;
use work.mupix_constants.all;
use work.mupix_registers.all;
use work.daq_constants.all;

entity mupix_ctrl is
    port(
        i_clk               : in  std_logic;
        i_reset_n           : in  std_logic;

        i_reg_add           : in  std_logic_vector(7 downto 0);
        i_reg_re            : in  std_logic;
        o_reg_rdata         : out std_logic_vector(31 downto 0);
        i_reg_we            : in  std_logic;
        i_reg_wdata         : in  std_logic_vector(31 downto 0);
    
        o_clock             : out std_logic_vector( 3 downto 0);
        o_SIN               : out std_logic_vector( 3 downto 0);
        o_mosi              : out std_logic_vector( 3 downto 0);
        o_csn               : out std_logic_vector(11 downto 0)--;
        );
end entity mupix_ctrl;

architecture RTL of mupix_ctrl is

    signal mp_fifo_clear            : std_logic;
    signal config_storage_input_data: std_logic_vector(32*5 + 31 downto 0);
    signal is_writing               : std_logic_vector(5 downto 0);
    signal is_writing_this_round    : std_logic_vector(5 downto 0);
    signal config_storage_write     : std_logic_vector(5 downto 0);
    signal enable_shift_reg_6       : std_logic_vector(5 downto 0);
    signal clk1                     : std_logic_vector(5 downto 0);
    signal clk2                     : std_logic_vector(5 downto 0);
    signal clk_step                 : std_logic_vector(3 downto 0);
    signal ld_regs                  : std_logic_vector(5 downto 0);

    signal rd_config                : std_logic;
    signal config_data              : std_logic_vector(5 downto 0);
    signal config_data29            : std_logic_vector(28 downto 0);

    signal slow_down                : std_logic_vector(15 downto 0);
    signal wait_cnt                 : std_logic_vector(15 downto 0);
    type mp_ctrl_state_type         is (idle, load_config, set_clks, writing, ld_spi_reg);
    signal mp_ctrl_state            : mp_ctrl_state_type;

begin


    e_mupix_reg_mapping : work.mupix_reg_mapping
    port map (
        i_clk156                    => i_clk,
        i_reset_n                   => i_reset_n,

        i_reg_add                   => i_reg_add,
        i_reg_re                    => i_reg_re,
        o_reg_rdata                 => o_reg_rdata,
        i_reg_we                    => i_reg_we,
        i_reg_wdata                 => i_reg_wdata,

        -- inputs  156--------------------------------------------

        -- outputs 156--------------------------------------------
        o_mp_ctrl_data              => config_storage_input_data,
        o_mp_fifo_write             => config_storage_write,
        o_mp_fifo_clear             => mp_fifo_clear,
        o_mp_ctrl_enable            => enable_shift_reg_6,
        o_mp_ctrl_slow_down(15 downto 0) => slow_down--,
    );

    e_mupix_ctrl_config_storage : work.mupix_ctrl_config_storage
    port map(
        i_clk                       => i_clk,
        i_reset_n                   => i_reset_n,
        i_clr_all                   => mp_fifo_clear,

        i_data                      => config_storage_input_data,
        i_wrreq                     => config_storage_write,

        o_data                      => config_data,
        o_is_writing                => is_writing,
        i_enable                    => enable_shift_reg_6,
        i_rdreq                     => rd_config--,
    );


    process(i_clk, i_reset_n)
        variable extra_round           : std_logic := '0';
    begin
        if(i_reset_n = '0')then
            wait_cnt                <= (others => '0');
            mp_ctrl_state           <= idle;
            config_data29           <= (others => '0');
            is_writing_this_round   <= (others => '0');
            clk1                    <= (others => '0');
            clk2                    <= (others => '0');
            clk_step                <= (others => '0');

        elsif(rising_edge(i_clk))then
            rd_config <= '0';
            wait_cnt  <= wait_cnt + 1;
            ld_regs(WR_CONF_BIT) <= '1'; -- bug in mp10, load config stays at 1#

            case mp_ctrl_state is
                when idle =>
                    extra_round         := '0';
                    
                    if(or_reduce(is_writing) = '1') then
                        mp_ctrl_state   <= load_config;
                        rd_config       <= '1';
                        wait_cnt        <= (others => '0');
                    end if;

                when load_config =>
                    config_data29                   <= (others => '0');
                    clk_step                        <= (others => '0');
                    extra_round                     := '0';
                    if(wait_cnt = x"0001") then
                        for I in 0 to 5 loop
                            config_data29(I*3)      <= config_data(I); -- TODO: check order !!
                        end loop;
                        is_writing_this_round       <= is_writing;
                        mp_ctrl_state               <= set_clks;
                    end if;

                when set_clks =>
                    clk_step    <= clk_step + 1;
                    for I in 0 to 5 loop
                        if(is_writing_this_round(I)='1') then
                            -- clocks (1.step 0 0, 2.step 1 0, 3.step 0 0, 4.step 0 1, 5.step 0 0)
                            case clk_step is
                                when x"0" => 
                                    config_data29(I*3 + 1) <= '0';
                                    config_data29(I*3 + 2) <= '0';
                                when x"1" => 
                                    config_data29(I*3 + 1) <= '1';
                                    config_data29(I*3 + 2) <= '0';
                                when x"2" => 
                                    config_data29(I*3 + 1) <= '0';
                                    config_data29(I*3 + 2) <= '0';
                                when x"3" => 
                                    config_data29(I*3 + 1) <= '0';
                                    config_data29(I*3 + 2) <= '1';
                                when x"4" => 
                                    config_data29(I*3 + 1) <= '0';
                                    config_data29(I*3 + 2) <= '0';
                                when x"5" =>
                                    if(is_writing(I)='0') then -- finished writing the complete reg this round --> additional load round
                                        config_data29(I + 18) <= '1'; -- set ld bit
                                        extra_round           := '1';
                                    end if;
                                when others =>
                                    
                            end case;
                        else
                            config_data29(I*3 + 1) <= '0';
                            config_data29(I*3 + 2) <= '0';
                        end if;
                    end loop;

                    if((extra_round = '1' or clk_step/=x"5") and (not clk_step=x"6")) then
                        mp_ctrl_state <= writing;
                    elsif(or_reduce(is_writing)='0') then -- done
                        mp_ctrl_state <= idle;
                    else                                  -- load next
                        mp_ctrl_state <= load_config;
                        wait_cnt      <= (others => '0');
                    end if;

                when writing =>
                 -- TODO: SPI write all bits in config_data29, then 
                    mp_ctrl_state   <= ld_spi_reg;
                    extra_round     := '0';
                when ld_spi_reg => -- TODO: CS mask
                    mp_ctrl_state   <= set_clks;
                when others =>
                    mp_ctrl_state <= idle;
            end case;
            
        end if;
    end process;

end RTL;
