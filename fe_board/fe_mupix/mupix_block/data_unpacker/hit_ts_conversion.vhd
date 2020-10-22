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
use work.datapath_components.all;
use work.mupix_constants.all;
use work.mupix_types.all;

entity hit_ts_conversion is 
    port (
        reset_n                 : in  std_logic;
        clk                     : in  std_logic;
        invert_TS               : in  std_logic;
        invert_TS2              : in  std_logic;
        gray_TS                 : in  std_logic;
        gray_TS2                : in  std_logic;
        hit_in                  : in  std_logic_vector(UNPACKER_HITSIZE-1 DOWNTO 0);
        hit_ena_in              : in  std_logic;
        hit_out                 : out std_logic_vector(UNPACKER_HITSIZE-1 DOWNTO 0);
        hit_ena_out             : out std_logic
    );
end hit_ts_conversion;

architecture rtl of hit_ts_conversion is

constant MAX_SIZE : integer := 16;

signal hit_in_TS        : std_logic_vector(MAX_SIZE-1 downto 0);
signal hit_in_TS2       : std_logic_vector(MAX_SIZE-1 downto 0);
signal hit_out_TS       : std_logic_vector(MAX_SIZE-1 downto 0);
signal hit_out_TS2      : std_logic_vector(MAX_SIZE-1 downto 0);

signal hit_in_r1        : std_logic_vector(UNPACKER_HITSIZE-1 DOWNTO 0);
signal hit_ena_in_r1    : std_logic;
signal hit_in_r2        : std_logic_vector(UNPACKER_HITSIZE-1 DOWNTO 0);
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
            TS_SIZE     <= TIMESTAMPSIZE_MP10;
            TS2_SIZE    <= CHARGESIZE_MP10;
            ts_temp     <= (others => '0');
            ts2_temp    <= (others => '0');
            ts_temp(TIMESTAMPSIZE_MP10-1 downto 0)  <= hit_in(15) & hit_in(9 downto 0); -- just reverting from unpacker
            ts2_temp(CHARGESIZE_MP10-1 downto 0)    <= hit_in(14 downto 10);
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
    i_degray_TS : gray_to_binary 
    generic map(MAXBITS => MAX_SIZE)
    port map(
        reset_n                 => reset_n,
        clk                     => clk,
        NBITS                   => TS_SIZE,
        gray_in                 => hit_in_TS,
        bin_out                 => hit_out_TS
    );

    i_degray_TS2 : gray_to_binary 
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
            hit_in_r1       <= (others => '0');
            hit_ena_in_r2   <= '0';
            hit_in_r2       <= (others => '0');

            hit_in_TS_reg   <= (others => '0');
            hit_in_TS2_reg  <= (others => '0');
        elsif(rising_edge(clk))then
            hit_in_r1       <= hit_in;
            hit_ena_in_r1   <= hit_ena_in;
            hit_in_r2       <= hit_in_r1;
            hit_ena_in_r2   <= hit_ena_in_r1;

            -- if gray decoding is not used, here we register the encoded, but optinally inverted timestamps
            hit_in_TS_reg   <= hit_in_TS;
            hit_in_TS2_reg  <= hit_in_TS2;
        end if;
    end process;

------- Output signals ----------------        

    output_sel: process(reset_n, clk)
    begin
        if(reset_n = '0')then
            hit_out         <= (others => '0');
            hit_ena_out     <= '0';
        elsif(rising_edge(clk))then
            hit_ena_out     <= hit_ena_in_r2;
            hit_out         <= hit_in_r2;

            if(gray_TS = '1')then
                hit_out(15)             <= hit_out_TS(TIMESTAMPSIZE_MP10-1);
                hit_out(9 downto 0)     <= hit_out_TS(TIMESTAMPSIZE_MP10-2 downto 0);
            else
                hit_out(15)             <= hit_in_TS_reg(TIMESTAMPSIZE_MP10-1);
                hit_out(9 downto 0)     <= hit_in_TS_reg(TIMESTAMPSIZE_MP10-2 downto 0);
            end if;

            if(gray_TS2 = '1')then
                hit_out(14 downto 10)   <= hit_out_TS2(CHARGESIZE_MP10-1 downto 0);
            else
                hit_out(14 downto 10)   <= hit_in_TS2_reg(CHARGESIZE_MP10-1 downto 0);
            end if;
        end if;
    end process;

end rtl;
