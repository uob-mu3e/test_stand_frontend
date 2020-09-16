----------------------------------------------------------------------------
-- I2C reading of firefly regs, FEB V2
-- Martin Mueller muellem@uni-mainz.de
----------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use work.firefly_constants.all;

ENTITY firefly_i2c is 
    generic(
        I2C_DELAY_g             : positive := 50000000--;
    );
    PORT(
        i_clk :             in  std_logic;
        i_reset_n :         in  std_logic;

        --I2C
        i_i2c_enable            : in    std_logic;
        o_Mod_Sel_n             : out   std_logic_vector(1 downto 0);
        io_scl                  : inout std_logic;
        io_sda                  : inout std_logic;
        i_int_n                 : in    std_logic_vector(1 downto 0);
        i_modPrs_n              : in    std_logic_vector(1 downto 0)--;
    );
END ENTITY;

architecture rtl of firefly_i2c is

    -- i2c signals --------------------------------------------
    signal i2c_rw               : std_logic;
    signal i2c_ena              : std_logic;
    signal i2c_busy             : std_logic;
    signal i2c_busy_prev        : std_logic;
    signal i2c_addr             : std_logic_vector(6 downto 0);
    signal i2c_data_rd          : std_logic_vector(7 downto 0);
    signal i2c_data_wr          : std_logic_vector(7 downto 0);
    type   i2c_state_type         is (idle, waiting1, i2cffly1, waiting2, i2cffly2);
    signal i2c_state            : i2c_state_type;
    signal i2c_data             : std_logic_vector(23 downto 0);
    signal i2c_counter          : unsigned(31 downto 0);

begin

    firefly_i2c_master: entity work.i2c_master
    generic map(
        input_clk   => 50_000_000,  --input clock speed from user logic in Hz
        bus_clk     => 400_000--,   --speed the i2c bus (scl) will run at in Hz
    )
    port map(
        clk         => i_clk,
        reset_n     => i_reset_n,
        ena         => i2c_ena,
        addr        => i2c_addr,
        rw          => i2c_rw,
        data_wr     => i2c_data_wr,
        busy        => i2c_busy,
        data_rd     => i2c_data_rd,
        ack_error   => open,
        sda         => io_sda,
        scl         => io_scl--,
    );

    process(i_clk, i_reset_n)
    variable busy_cnt           : integer := 0;
    begin
        if(i_reset_n = '0') then
            i2c_state       <= idle;
            i2c_ena         <= '0';
            o_Mod_Sel_n     <= "11";
            i2c_rw          <= '1';
            i2c_busy_prev   <= '0';
            i2c_counter     <= (others => '0');
            
        elsif(rising_edge(i_clk)) then
            case i2c_state is
                when idle =>
                    o_Mod_Sel_n     <= "11";
                    i2c_counter     <= (others => '0');
                    if(i_i2c_enable = '1') then 
                        i2c_state       <= waiting1;
                    end if;
                when waiting1 =>
                    o_Mod_Sel_n(0)  <= '0'; -- want to talk to firefly 1 (active low)
                    o_Mod_Sel_n(1)  <= '1';
                    i2c_counter     <= i2c_counter + 1;
                    
                    if(i2c_counter = I2C_DELAY_g) then -- wait for assert time of mod_sel (a few hundred ms)
                        i2c_state       <= i2cffly1;
                    end if;
                    
                when i2cffly1 => -- i2c transaction with firefly 1
                    i2c_busy_prev   <= i2c_busy;
                    i2c_counter     <= (others => '0');
                    if(i2c_busy_prev = '0' AND i2c_busy = '1') then
                        busy_cnt := busy_cnt + 1;
                    end if;
                    
                    case busy_cnt is
                        when 0 =>
                            i2c_ena     <= '1';
                            i2c_addr    <= FFLY_DEV_ADDR_7;
                            i2c_rw      <= '0'; -- 0: write, 1: read
                            i2c_data_wr <= ADDR_TEMPERATURE;
                        when 1 =>
                            i2c_rw      <= '1';
                        when 2 =>
                            i2c_rw      <= '0';
                            i2c_data_wr <= RX1_PWR_1;
                            if(i2c_busy = '0') then
                                i2c_data(7 downto 0) <= i2c_data_rd; -- read data from busy_cnt = 1
                            end if;
                        when 3 =>
                            i2c_rw      <= '1';
                        when 4 =>
                            i2c_rw      <= '0';
                            i2c_data_wr <= RX1_PWR_2;
                            if(i2c_busy = '0') then
                                i2c_data(15 downto 8) <= i2c_data_rd; -- read data from busy_cnt = 1
                            end if;
                        when 5 =>
                            i2c_rw      <= '1';
                        when 6 =>
                            i2c_ena     <= '0';
                            if(i2c_busy = '0') then
                                i2c_data(23 downto 16)  <= i2c_data_rd;
                                busy_cnt                := 0;
                                i2c_state               <= waiting2;
                            end if;
                        when others => null;
                    end case;
                    
                when waiting2 =>
                    o_Mod_Sel_n(1)  <= '0';
                    o_Mod_Sel_n(0)  <= '1';
                    i2c_state       <= i2cffly2;
                    i2c_counter     <= i2c_counter + 1;
                    
                    if(i2c_counter = I2C_DELAY_g) then -- wait for assert time of mod_sel (a few hundred ms)
                        i2c_state       <= i2cffly1;
                    end if;
                when i2cffly2 => -- i2c transaction with firefly 2
                --todo: insert same thing when ffly1 is working
                    i2c_state       <= idle;
                    i2c_counter     <= (others => '0');
                    
                when others =>
                    i2c_state       <= idle;
            end case;
        end if;
    end process;

end architecture;