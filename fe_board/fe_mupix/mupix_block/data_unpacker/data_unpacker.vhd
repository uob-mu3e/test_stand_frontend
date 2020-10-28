-------------------------------------------------------------
--
-- This is the Data unpacker for MuPix10 in the online repo 
-- THIS IS NOT IDENTICAL TO MUPIX8_DAQ Data_unpacker !!!
-- THIS IS NOT THE DETECTORFPGA Data_unpacker !!!
--
-- Martin Mueller, Oktober 2020
-- muellem@uni-mainz.de
-- 
-- derived from mupix8_daq data_unpacker
-- (mp10 only, single hit_out format without coarsecounters)
-- Sebastian Dittmeier, September 2017
-- Ann-Kathrin Perrevoort, April 2017
--------------------------------------------------------------


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.mupix_constants.all;
use work.daq_constants.all;



entity data_unpacker is 
    generic (
        COARSECOUNTERSIZE   : integer   := 32
    );
    port (
        reset_n             : in  std_logic;
        clk                 : in  std_logic;
        datain              : in  std_logic_vector (7 downto 0);
        kin                 : in  std_logic;
        readyin             : in  std_logic;
        hit_out             : out std_logic_vector (31 downto 0); -- Link[7:0] & Row[7:0] & Col[7:0] & Charge[5:0] & TS[9:0]
        hit_ena             : out std_logic;
        coarsecounter       : out std_logic_vector (COARSECOUNTERSIZE-1 downto 0); -- Gray Counter[7:0] & Binary Counter [23:0]
        coarsecounter_ena   : out std_logic;
        link_flag           : out std_logic;
        errorcounter        : out std_logic_vector(31 downto 0)
    );
end data_unpacker;

architecture RTL of data_unpacker is

    type state_type is (IDLE, ERROR, COUNTER, LINK, DATA);
    signal NS                   : state_type;

    signal data_i               : std_logic_vector(31 downto 0) := (others => '0');

    signal errorcounter_reg     : std_logic_vector(31 downto 0);

    signal link_i               : std_logic_vector(7 downto 0);
    signal link_reg             : std_logic_vector(7 downto 0);
    signal link_toggle          : std_logic := '0';
    signal data_mode            : std_logic;

    signal cnt4                 : std_logic_vector(1 downto 0);

    signal counter_seen         : std_logic;
    signal coarse_reg           : std_logic;
    signal hit_reg              : std_logic;
    signal link_flag_reg        : std_logic;

begin

    errorcounter     <= errorcounter_reg;

    fsmProc: process (clk, reset_n)
    begin
        if reset_n = '0' then
            NS <= IDLE;

            hit_ena             <= '0';
            hit_out             <= (others => '0');
            coarsecounter       <= (others => '0');
            coarsecounter_ena   <= '0';
            link_flag           <= '0';

            link_flag_reg       <= '0';
            coarse_reg          <= '0';
            hit_reg             <= '0';

            link_i              <= (others =>'0');
            link_reg            <= (others =>'0');
            link_toggle         <= '0';

            errorcounter_reg    <= (others => '0');

            cnt4                <= "00";
            data_mode           <= '0'; -- indicates if all counter mode or actual hit data
            counter_seen        <= '0';
        elsif rising_edge(clk) then

            link_flag_reg       <= '0';
            coarse_reg          <= '0';
            hit_reg             <= '0';

            hit_ena             <= hit_reg;
            coarsecounter_ena   <= coarse_reg;
            link_flag           <= link_flag_reg;

            coarsecounter       <= data_i(7 downto 0) & data_i(31 downto 8); -- gray counter & binary counter
            if(hit_reg = '1')then
                -- The data arrives from MuPix10: TS2[4:0] TS1[10:0] Col[6:0] Row[8:0]
                    -- TS2 = data_i(31 downto 27)
                    -- TS1 = data_i(26 downto 16)
                    -- Col = data_i(15 downto 9)
                    -- Row = data_i(8 downto 0)
                -- We present to the outside:
                    -- hit_out(31 downto 24): row
                    -- hit_out(23 downto 16): col
                    -- hit_out(15 downto 10): ts2
                    -- hit_out(9 downto 0)    : ts1
                hit_out         <= data_i(7 downto 0) & data_i(8) & data_i(15 downto 9) & -- Row(7:0) & Row(8) & Col(7b)
                                    data_i(26) & data_i(31 downto 27) & data_i(25 downto 16);         -- TS1(10)  & TS2(5b)  & TS1(9:0)
            end if;

            if(readyin = '0')then
                link_reg        <= (others => '0');
                link_toggle     <= '0';
                NS              <= IDLE;
            else
                data_i          <= data_i(23 downto 0) & datain;

                    case NS is
                    
                        when IDLE =>
                            cnt4            <= "00";
                            link_toggle     <= '0';
                            if kin = '0'then                                   -- counter mode
                                if(data_mode = '1' and counter_seen = '1')then -- we expect valid hit data here after LINK ID and counter was seen
                                    NS      <= DATA;
                                    cnt4    <= "01";
                                else
                                    NS      <= COUNTER;
                                    cnt4    <= "01";
                                end if;
                            elsif kin = '1' and datain = k28_0 then -- data mode
                                NS <= LINK;
                            elsif kin = '1' and datain = k28_5 then
                                NS <= IDLE; 
                            else
                                NS <= ERROR;
                            end if;


                        when COUNTER =>
                            if kin = '0' then
                                cnt4                <= cnt4 + '1';
                                if cnt4 = "11" then
                                    coarse_reg      <= '1';
                                    counter_seen    <= '1';
                                    NS              <= IDLE;
                                end if;
                            elsif kin = '1' and datain = k28_5 then    --and counter_int = 3 then
                                NS <= IDLE;
                            else
                                NS <= ERROR;
                            end if;

                        when LINK =>
                            if kin = '0' then
                                if link_toggle = '0' then
                                    link_reg            <= datain;
                                elsif link_reg = datain then
                                    link_i              <= link_reg;
                                    link_flag_reg       <= '1';
                                    link_toggle         <= '0';
                                    NS                  <= IDLE;
                                    data_mode           <= '1';
                                    counter_seen        <= '0';
                                else
                                    NS                <= ERROR;
                                end if;
                            elsif kin = '1' and datain = k28_0 then
                                link_toggle <= '1';
                            else
                                NS <= ERROR;
                            end if;

                        when DATA =>
                        
                            if kin = '0' then
                                cnt4        <= cnt4 + '1';
                                if cnt4 = "11" then
                                    hit_reg <= '1';
                                    NS      <= IDLE;
                                end if;
                            elsif kin = '1' and datain = k28_5 then
                                NS          <= IDLE;
                            else
                                NS          <= ERROR;
                            end if;

                        when ERROR =>
                            errorcounter_reg <= errorcounter_reg + '1';
                            if ( kin = '1' and datain = k28_5 ) then
                                NS <= IDLE;
                            else
                                NS <= ERROR;
                            end if;
                        
                        when others =>
                            NS <= IDLE;
                    end case; -- NS
            end if; -- readyin

        end if;

    end process;

end RTL;
