-- Convert gray code to a binary code
-- Sebastian Dittmeier 
-- September 2017
-- dittmeier@physi.uni-heidelberg.de
-- based on code by Niklaus Berger
--
-- takes 3 clock cycles now to do the full thing


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;


entity hit_ts_conversion is 
    port (
        reset_n     : in  std_logic;
        clk         : in  std_logic;
        invert_TS   : in  std_logic;
        invert_TS2  : in  std_logic;
        gray_TS     : in  std_logic;
        gray_TS2    : in  std_logic;

        o_ts        : out std_logic_vector(10 downto 0);
        o_row       : out std_logic_vector(7 downto 0);
        o_col       : out std_logic_vector(7 downto 0);
        o_ts2       : out std_logic_vector(4 downto 0);
        o_hit_ena   : out std_logic;
        
        i_ts        : in  std_logic_vector(10 downto 0);
        i_row       : in  std_logic_vector(7 downto 0);
        i_col       : in  std_logic_vector(7 downto 0);
        i_ts2       : in  std_logic_vector(4 downto 0);
        i_hit_ena   : in  std_logic--;
        
    );
end hit_ts_conversion;

architecture rtl of hit_ts_conversion is

constant MAX_SIZE : integer := 16;

signal hit_in_TS        : std_logic_vector(MAX_SIZE-1 downto 0);
signal hit_in_TS2       : std_logic_vector(MAX_SIZE-1 downto 0);
signal hit_out_TS       : std_logic_vector(MAX_SIZE-1 downto 0);
signal hit_out_TS2      : std_logic_vector(MAX_SIZE-1 downto 0);

signal row_r1           : std_logic_vector(7 downto 0);
signal row_r2           : std_logic_vector(7 downto 0);
signal col_r1           : std_logic_vector(7 downto 0);
signal col_r2           : std_logic_vector(7 downto 0);

signal hit_ena_in_r1    : std_logic;
signal hit_ena_in_r2    : std_logic;
signal hit_in_TS_reg    : std_logic_vector(MAX_SIZE-1 downto 0);
signal hit_in_TS2_reg   : std_logic_vector(MAX_SIZE-1 downto 0);

signal ts_temp          : std_logic_vector(MAX_SIZE-1 downto 0);
signal ts2_temp         : std_logic_vector(MAX_SIZE-1 downto 0);
signal TS_SIZE          : integer range 0 to MAX_SIZE;
signal TS2_SIZE         : integer range 0 to MAX_SIZE;

begin


------- Input selection ------------------
    input_chip: process (clk, reset_n)
    begin
        if reset_n = '0' then
            TS_SIZE     <= 0;
            TS2_SIZE    <= 0;
            ts_temp     <= (others => '0');
            ts2_temp    <= (others => '0');

        elsif(rising_edge(clk))then
            TS_SIZE     <= work.mupix.TIMESTAMPSIZE;
            TS2_SIZE    <= work.mupix.CHARGESIZE_MP10;
            ts_temp     <= (others => '0');
            ts2_temp    <= (others => '0');
            ts_temp(work.mupix.TIMESTAMPSIZE-1 downto 0)  <= i_ts;
            ts2_temp(work.mupix.CHARGESIZE_MP10-1 downto 0)    <= i_ts2;
        end if;
    end process;

------- Optional inversion before decoding ----------
    with invert_TS select hit_in_TS <=
        not ts_temp    when '1',
        ts_temp        when others;

    with invert_TS2 select hit_in_TS2 <=
        not ts2_temp    when '1',
        ts2_temp        when others;

------- Gray Decoding ----------------
    i_degray_TS :entity work.gray_to_binary 
    generic map(MAXBITS => MAX_SIZE)
    port map(
        reset_n                 => reset_n,
        clk                     => clk,
        NBITS                   => TS_SIZE,
        gray_in                 => hit_in_TS,
        bin_out                 => hit_out_TS
    );

    i_degray_TS2 :entity work.gray_to_binary 
    generic map(MAXBITS => MAX_SIZE)
    port map(
        reset_n                 => reset_n,
        clk                     => clk,
        NBITS                   => TS2_SIZE,
        gray_in                 => hit_in_TS2,
        bin_out                 => hit_out_TS2
    );

------- Pipelining inputs --------
-- this process happens concurrently with the input selection 
-- then decoding also takes 1 clock cyle
-- so two pipelines
    pipelining: process(reset_n, clk)
    begin
        if(reset_n = '0')then
            hit_ena_in_r1   <= '0';
            hit_ena_in_r2   <= '0';
            col_r1          <= (others => '0');
            col_r2          <= (others => '0');
            row_r1          <= (others => '0');
            row_r2          <= (others => '0');
            
            hit_in_TS_reg   <= (others => '0');
            hit_in_TS2_reg  <= (others => '0');
        elsif(rising_edge(clk))then
            hit_ena_in_r1   <= i_hit_ena;
            hit_ena_in_r2   <= hit_ena_in_r1;
            col_r1          <= i_col;
            col_r2          <= col_r1;
            row_r1          <= i_row;
            row_r2          <= row_r1;

            -- if gray decoding is not used, here we register the encoded, but optinally inverted timestamps
            hit_in_TS_reg   <= hit_in_TS;
            hit_in_TS2_reg  <= hit_in_TS2;
        end if;
    end process;

------- Output signals ----------------        

    output_sel: process(reset_n, clk)
    begin
        if(reset_n = '0')then
            o_hit_ena       <= '0';
            o_col           <= (others => '0');
            o_row           <= (others => '0');
            o_ts            <= (others => '0');
            o_ts2           <= (others => '0');
        elsif(rising_edge(clk))then
            o_hit_ena       <= hit_ena_in_r2;
            o_col           <= col_r2;
            o_row           <= row_r2;

            if(gray_TS = '1')then
                o_ts        <= hit_out_TS(work.mupix.TIMESTAMPSIZE-1 downto 0);
            else
                o_ts        <= hit_in_TS_reg(work.mupix.TIMESTAMPSIZE-1 downto 0);
            end if;

            if(gray_TS2 = '1')then
                o_ts2       <= hit_out_TS2(work.mupix.CHARGESIZE_MP10-1 downto 0);
            else
                o_ts2       <= hit_in_TS2_reg(work.mupix.CHARGESIZE_MP10-1 downto 0);
            end if;
        end if;
    end process;

end rtl;
