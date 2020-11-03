library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity max10_spi_main is
generic(
    SS      : std_logic; -- '1' is chip selected
    Del     : integer := 1; -- time delay
    R       : std_logic; -- when (r/w = R) => read || when (r/w != R) => write
    lanes   : integer := 4--; --2 or 4 is possible
);
port(
    -- LED
    o_led       : out std_logic_vector(7 downto 0);
    -- clk & reset
    i_clk_50    : in  std_logic;
    i_clk_100   : in  std_logic;
    i_reset_n   : in  std_logic;
    i_command	: in  std_logic_vector(15 downto 0); --[15-9] empty ,[8-2] cnt , [1] rw , [0] aktiv, 
    ------ Arria Data --register interface 
    o_Ar_rw		: out std_logic;
    o_Ar_data	: out std_logic_vector(31 downto 0);
    o_Ar_addr_o	: out std_logic_vector(6 downto 0);
    o_Ar_done	: out std_logic;
    i_Max_data	: in  std_logic_vector(31 downto 0);
    i_Max_addr	: in  std_logic_vector(15 downto 0);
    -- Max10 SPI
    o_SPI_cs    : out std_logic;
    o_SPI_clk   : out std_logic;
    io_SPI_mosi : inout std_logic;
    io_SPI_miso : inout std_logic;
    io_SPI_D1   : inout std_logic;
    io_SPI_D2   : inout std_logic;
    io_SPI_D3   : inout std_logic--;

);
end entity;

architecture rtl of max10_spi_main is

    type State_type is (Idle, Adrr , W_Data , Deley , R_Data );
    type s_regs is Array (0 to (lanes-1)) of std_logic_vector(((32/lanes)-1) downto 0);
    
    signal State : State_type;
    --imput signal -> vector for 2/4 lanes
	signal io_SPI_D    : std_logic_vector(3 downto 0);
	
	--shift register

	signal s_r_reg_16	: s_regs ; -- read SPI Slave
	signal s_reg_16		: s_regs ; -- write SPI Slave
	
	signal cnt_8		: integer range 0 to (16/lanes) ;    -- addr  cnt
	signal cnt_del		: integer range 0 to 64;             -- delay cnt
	signal cnt_16		: integer range -1 to (32/lanes)+1 := 0 ; -- read cnt --BUG muss warum kommt er über 16 mit dem cnt ?
	signal cnt_words	: unsigned (6 downto 0):="0000000";
	signal cnt_words_test : std_logic_vector(6 downto 0);
	
	signal addr_offset	: unsigned(6 downto 0);
	
	signal aktiv		: std_logic;
	signal rw 			: std_logic;
	signal ss_del		: std_logic;
	signal clk100_del   : std_logic; -- read 100 clk delay
	
	------ Aria Register -----
	
--    signal i_command	:   std_logic_vector(15 downto 0); --[15-9] empty ,[8-2] cnt , [1] rw , [0] aktiv, 
	
	------ Aria Data --register interface 
--	signal o_Ar_rw		:  std_logic;
--	signal o_Ar_data	:  std_logic_vector(31 downto 0);
--	signal o_Ar_addr_o	:  std_logic_vector(6 downto 0);
	signal o_Ar_done_s	:  std_logic;
--	
--	signal i_Max_data	:   std_logic_vector(31 downto 0);
--	signal i_Max_addr	:   std_logic_vector(15 downto 0); --[15-8] empty , [7] RW , [6-0] addr
	
	----- SPI interface for simulation---
	--signal o_SPI_cs    : std_logic;
	--signal o_SPI_clk   : std_logic;
	--signal o_SPI_mosi  : std_logic;
	--signal i_SPI_miso  : std_logic;
	--signal io_SPI_D1   : std_logic;
	--signal io_SPI_D2   : std_logic;
	--signal io_SPI_D3   : std_logic;

begin
	o_Ar_done <= o_Ar_done_s;
    
	-- 2/4 lane Imput
	io_SPI_D(0) <= io_SPI_mosi;
	io_SPI_D(1) <= io_SPI_D1;
	io_SPI_D(2) <= io_SPI_D2;
	--io_SPI_D(3) <= io_SPI_D3;
	io_SPI_D(3) <= io_SPI_miso;
	
	--commands	
	aktiv 		<= i_command(0);
	rw 			<= i_command(1);
	
	--o_SPI_cs 	<=  not SS when State = Idle else SS;
	o_SPI_clk 	<= i_clk_50 when State /= Idle else '0';
	
	dual : if lanes = 2 generate
		io_SPI_mosi <= s_reg_16(0)((32/lanes)-1) when State = Adrr or State = W_Data else 'Z';
		io_SPI_D1	<= s_reg_16(1)((32/lanes)-1) when State = Adrr or State = W_Data else 'Z';
        io_SPI_D2	<= 'Z';
		io_SPI_miso	<= 'Z';
	end generate;
	
	quad : if lanes = 4 generate
		io_SPI_mosi <= s_reg_16(0)((32/lanes)-1) when State = Adrr or State = W_Data else 'Z';
		io_SPI_D1	<= s_reg_16(1)((32/lanes)-1) when State = Adrr or State = W_Data else 'Z';
		io_SPI_D2	<= s_reg_16(2)((32/lanes)-1) when State = Adrr or State = W_Data else 'Z';
		io_SPI_miso	<= s_reg_16(3)((32/lanes)-1) when State = Adrr or State = W_Data else 'Z';
	end generate;
	
process(i_clk_100)
begin
    
    if(rising_edge(i_clk_100)) then --BUG sammelt immer ? 
        for i in 0 to lanes-1 loop
            s_r_reg_16(i)(0) <= io_SPI_D(i);
        end loop;
    end if;
end process;

process(i_clk_50 , i_reset_n)
begin

    if (i_reset_n = '0') then
        State <= Idle;
    
    elsif(rising_edge(i_clk_50)) then
    
    o_Ar_addr_o <= std_logic_vector(addr_offset);
    
        if ( aktiv = '0' and State /= Idle ) then
            State <= Idle;
        else
            CASE State is
                
                When Idle =>
                    o_Ar_done_s   <= '0';
                    o_Ar_rw     <= R ; 
                    o_SPI_cs    <= not SS;
                    addr_offset <= "0000000" ;
                    
                    if aktiv = '1' and o_Ar_done_s = '0' then
                        for i in  0 to (lanes -1) loop
                            s_reg_16(i)((32/lanes)-1 downto (16/lanes)) <= i_Max_addr((15-16/lanes*i) downto (16-16/lanes*(i+1)));
                        end loop;
                        o_SPI_cs <= SS ;
                        State 	<= Adrr;
                        cnt_16 	<= 1;
                        cnt_8 	<= 1;
                        cnt_words  	<= unsigned(i_command(8 downto 2));
                    end if;
                
                When Adrr => 
                    if cnt_8 /= (16/lanes) then
                        cnt_8 <= cnt_8 + 1;
                        for i in  0 to (lanes -1) loop
                            for j in  1 to (32/lanes)-1 loop
                                s_reg_16(i)(j) <= s_reg_16(i)(j-1);
                            end loop;
                        end loop;
                    elsif rw = R then
                        State <= Deley;
                    else
                        State <= W_Data;
                        for i in  0 to (lanes -1) loop
                            s_reg_16(i) <= i_Max_data((31-32/lanes*i) downto (32-32/lanes*(i+1)));
                        end loop;
                        addr_offset <= addr_offset + 1 ;
                    end if;	
                    
                When W_Data =>
                    o_Ar_rw     <= R ;
                    cnt_16 <= cnt_16 +1;
                    for i in  0 to (lanes -1) loop
                        for j in  1 to (32/lanes)-1 loop
                            s_reg_16(i)(j) <= s_reg_16(i)(j-1);
                        end loop;
                    end loop;
                    if cnt_16 = (32/lanes) then
                        cnt_16 <= 1;
                        cnt_words <= cnt_words -1;
                        addr_offset <= addr_offset + 1 ;
                        for i in  0 to (lanes -1) loop
                            s_reg_16(i) <= i_Max_data((31-32/lanes*i) downto (32-32/lanes*(i+1)));
                        end loop;				
                        if cnt_words = 0 then
                            State <= Deley;
                            o_Ar_done_s <= '1';
                        end if;
                    end if;
                
                When Deley =>
                    if cnt_del /= Del then
                        cnt_del <= cnt_del +1;
                    elsif (o_Ar_done_s ='1') then
                        State <= Idle;
                        o_Ar_done_s <= '1';
                    else
                        State <= R_Data;
                        cnt_16 <= -1;
    --                     for i in 0 to lanes-1 loop
    --                         s_r_reg_16(i)(0) <= io_SPI_D(i);
    --                     end loop;
                    end if;
                    
                When R_Data =>
                    o_Ar_rw     <= R ;
                    cnt_16 <= cnt_16 +1;
                    --s_r_reg_16(0)(0) <= i_SPI_miso;
    -- 				for i in 0 to lanes-1 loop
    --                     s_r_reg_16(i)(0) <= io_SPI_D(i);
    -- 				end loop;
                    
                    for i in  0 to (lanes -1) loop
                        for j in  1 to (32/lanes)-1 loop
                            s_r_reg_16(i)(j) <= s_r_reg_16(i)(j-1);
                        end loop;
                    end loop;
                    
                    
                    if cnt_16 = (32/lanes) then
                        o_Ar_rw     <= not R ;
                        cnt_words <= cnt_words -1;
                        for i in  0 to (lanes -1) loop
                            o_Ar_data((31-32/lanes*i) downto (32-32/lanes*(i+1))) <= s_r_reg_16(i);
                        end loop;

                        addr_offset <= addr_offset + 1 ;
                        cnt_16 <= 1;
                        if cnt_words = 0 then
                        
                            State <= Deley; 
                            o_Ar_done_s <= '1';
                    
                        end if;		
                    end if;
                    
                When others =>
                    State <= Idle;
                    
            end CASE;
        end if;
	end if;
end process;


---- SPI TEST ----

--    e_arria_reg_tb : entity work.arria_reg_tb
--    generic map ( R => R)
--    port map(
--        reset_n =>   i_reset_n,
--        i_clk   =>   i_clk_50,
--        
--        i_adrr  =>   o_Ar_addr_o, -- Base addr = 0
--        i_data  =>   o_Ar_data,
--        
--       i_R     =>   o_Ar_rw,
--        i_W     =>   '0',
--        
--        i_SPI_c =>   o_Ar_done, -- falls der SPI über register gesteuert werden soll.
--        o_SPI_c =>   i_command,
--        o_SPI_a =>   i_Max_addr,
--        
--        o_data  =>   i_Max_data--,

--    );
    
    --SPI_Slave : entity work.SPI_Slave  -- for simulation
    --port map (
    
    --i_SPI_cs	=> o_SPI_cs,
	--i_SPI_clk	=> o_SPI_clk,
	--i_SPI_mosi	=> o_SPI_mosi,
	--o_SPI_miso	=> i_SPI_miso,
	--io_SPI_D1	=> io_SPI_D1,
	--io_SPI_D2	=> io_SPI_D2,
	--io_SPI_D3	=> io_SPI_D3--;
    
    --);

---- SPI TEST ----


end rtl;
