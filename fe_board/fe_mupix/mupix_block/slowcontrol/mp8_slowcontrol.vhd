----------------------------------------------------------------------------
-- Slow Control Unit for MuPix8
--
-- Sebastian Dittmeier, Heidelberg University
-- dittmeier@physi.uni-heidelberg.de
--
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

entity mp8_slowcontrol is
    port(
        clk:                in  std_logic;
        reset_n:            in  std_logic;
        ckdiv:              in  std_logic_vector(15 downto 0);
        mem_data:           in  std_logic_vector(31 downto 0);
        wren:               in  std_logic;
        ld_in:              in  std_logic;
        rb_in:              in  std_logic;
        ctrl_dout:          in  std_logic;
        ctrl_din:           out std_logic;
        ctrl_clk1:          out std_logic;
        ctrl_clk2:          out std_logic;
        ctrl_ld:            out std_logic;
        ctrl_rb:            out std_logic;
        busy_n:             out std_logic := '0';   -- default value
        dataout:            out std_logic_vector(31 downto 0)
        );
end entity mp8_slowcontrol;


architecture rtl of mp8_slowcontrol is

type state_type is (waiting, rb_hi, data_in, clock_1_hi, clock_1_lo, clock_2_hi, clock_2_lo);
signal state : state_type;

signal ckdiv_reg    : std_logic_vector(15 downto 0);
signal cyclecounter : std_logic_vector(4 downto 0); 
signal mem_data_reg : std_logic_vector(31 downto 0);
signal wren_last    : std_logic;
signal dataout_reg  : std_logic_vector(31 downto 0);

signal ld_in_last   : std_logic;
signal rb_in_last   : std_logic;

begin


process(clk, reset_n)

begin
if(reset_n = '0') then
    ctrl_ld     <= '0';
    ctrl_rb     <= '0';
    ctrl_din    <= '0';
    ctrl_clk1   <= '0';
    ctrl_clk2   <= '0';
    busy_n      <= '0';
    dataout     <= (others => '0');
    dataout_reg <= (others => '0');
    wren_last   <= '0';
    ld_in_last  <= '0';
    rb_in_last  <= '0';
    ckdiv_reg   <= (others => '0');
    state       <= waiting;

    mem_data_reg <= (others => '0');
    cyclecounter <= (others => '0');

elsif(clk'event and clk = '1') then

    case state is

        when waiting =>	
            busy_n      <= '1';             -- indicate that no communication is in progress
            cyclecounter<= (others => '0'); 
            wren_last   <= wren;            -- we do not write again directly after data has been written
            ckdiv_reg   <= (others => '0');
            ctrl_din    <= '0';
            ctrl_clk1   <= '0';
            ctrl_clk2   <= '0';
            ctrl_ld     <= '0';
            ctrl_rb     <= '0';
            dataout_reg <= dataout_reg;
            dataout     <= dataout_reg;
            ld_in_last  <= ld_in;
            rb_in_last  <= rb_in;
            if(ld_in = '1' and ld_in_last = '0')then
                cyclecounter    <= (others => '1');
                ctrl_ld         <= '1';
                state           <= clock_2_lo;
                busy_n          <= '0';
            elsif(rb_in = '1' and rb_in_last = '0')then
                cyclecounter    <= (others => '1');
                ctrl_rb         <= '1';
                state           <= rb_hi;
                busy_n          <= '0';
            elsif(wren = '1' and wren_last = '0')then
                mem_data_reg    <= mem_data;    -- data is copied into shiftregister
                state           <= data_in;
                busy_n          <= '0';
            end if;
            
        when rb_hi  =>
            ckdiv_reg       <= ckdiv_reg + '1'; -- clock division
            if(ckdiv_reg >= ckdiv) then
                ckdiv_reg   <= (others => '0');
                state       <= clock_1_hi;
            end if;
            
        when data_in =>
            ctrl_din        <= mem_data_reg(31);-- put the MSB on the line
            ckdiv_reg       <= ckdiv_reg + '1'; -- clock division
            if(ckdiv_reg >= ckdiv) then
                ckdiv_reg   <= (others => '0');
                state       <= clock_1_hi;
            end if;
            
        when clock_1_hi =>
            ctrl_clk1       <= '1';
            ckdiv_reg       <= ckdiv_reg + '1'; -- clock division
            if(ckdiv_reg >= ckdiv) then
                ckdiv_reg   <= (others => '0');
                state       <= clock_1_lo;
                if(rb_in_last = '0')then -- sample only if we do not clock the readback bit!
                    dataout_reg <= dataout_reg(30 downto 0) & ctrl_dout; -- does not really matter when it is sampled
                end if;
            end if;
            
        when clock_1_lo =>
            ctrl_clk1       <= '0';
            ckdiv_reg       <= ckdiv_reg + '1'; -- clock division
            if(ckdiv_reg >= ckdiv) then
                ckdiv_reg   <= (others => '0');
                state       <= clock_2_hi;
            end if;
            
        when clock_2_hi =>
            ctrl_clk2       <= '1';
            ckdiv_reg       <= ckdiv_reg + '1'; -- clock division
            if(ckdiv_reg >= ckdiv) then
                ckdiv_reg   <= (others => '0');
                state       <= clock_2_lo;
            end if;
            
        when clock_2_lo =>
            ctrl_clk2        <= '0';
            ckdiv_reg        <= ckdiv_reg + '1'; -- clock division
            if(ckdiv_reg >= ckdiv) then
                ckdiv_reg    <= (others => '0');
                cyclecounter <= cyclecounter + '1';
                mem_data_reg <= mem_data_reg(30 downto 0) & '0';
                if(cyclecounter < "11111")then
                    state    <= data_in;
                else
                    state    <= waiting;
                end if;
            end if;
        
        when others =>
            state   <= waiting;
            
    end case;
end if;
end process;

end rtl;