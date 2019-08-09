library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
--use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
use std.textio.all;
use IEEE.std_logic_textio.all; 


--  A testbench has no ports.
entity mupix_block_tb is
end mupix_block_tb;

architecture behav of mupix_block_tb is
  --  Declaration of the component that will be instantiated.
	component mupix_block is
    generic(
        NCHIPS : integer :=4
    );
    port (
    
        -- chip dacs
        i_CTRL_SDO_A    : in std_logic;
        o_CTRL_SDI_A    : out std_logic;
        o_CTRL_SCK1_A   : out std_logic;
        o_CTRL_SCK2_A   : out std_logic;
        o_CTRL_Load_A   : out std_logic;
        o_CTRL_RB_A     : out std_logic;
        i_data_chip_dacs : in std_logic_vector(31 downto 0);
        o_add_chip_dacs : out std_logic_vector(15 downto 0);
        
        -- board dacs
        i_SPI_DOUT_ADC_0_A      : in std_logic;
        o_SPI_DIN0_A            : out std_logic;
        o_SPI_CLK_A             : out std_logic;
        o_SPI_LD_ADC_A          : out std_logic;
        o_SPI_LD_TEMP_DAC_A     : out std_logic;
        o_SPI_LD_DAC_A          : out std_logic;
        o_add_board_dacs        : out std_logic_vector(3 downto 0);
        i_data_board_dacs       : in std_logic_vector(31 downto 0);
        o_data_board_dacs       : out std_logic_vector(31 downto 0);
        o_wen_data_board_dacs   : out std_logic;
        
        i_ckdiv         : in std_logic_vector(15 downto 0);

        i_reset         : in std_logic;
        -- 156.25 MHz
        i_clk           : in std_logic;
        i_clk125        : in std_logic--;
    );
	end component mupix_block;
	
	component ip_ram is
    Port ( clock   : in  STD_LOGIC;
           wren : in  STD_LOGIC;
           wraddress   : in  STD_LOGIC_VECTOR (7 downto 0);
           rdaddress   : in  STD_LOGIC_VECTOR (7 downto 0);
           data   : in  STD_LOGIC_VECTOR (31 downto 0);
           q  : out STD_LOGIC_VECTOR (31 downto 0)
         );
    end component ip_ram;
    


  --  Specifies which entity is bound with the component.
  		
  		signal clk : std_logic;
  		signal reset_n : std_logic := '1';
  		signal reset : std_logic;
  		
  		signal i_CTRL_SDO_A : std_logic;
  		signal o_CTRL_SDI_A : std_logic;
  		signal o_CTRL_SCK1_A : std_logic;
  		signal o_CTRL_SCK2_A : std_logic;
  		signal o_CTRL_Load_A : std_logic;
  		signal o_CTRL_RB_A : std_logic;
  		signal i_data_chip_dacs : std_logic_vector(31 downto 0);
  		signal o_add_chip_dacs : std_logic_vector(15 downto 0);
  		signal wadd_chip : std_logic_vector(15 downto 0);
  		signal wdata_chip : std_logic_vector(31 downto 0);
  		signal wchip : std_logic;
  		signal chip_state : std_logic_vector(3 downto 0);
  		
  		signal i_SPI_DOUT_ADC_0_A : std_logic;
  		signal o_SPI_DIN0_A : std_logic;
  		signal o_SPI_CLK_A : std_logic;
  		signal o_SPI_LD_ADC_A : std_logic;
  		signal o_SPI_LD_TEMP_DAC_A : std_logic;
  		signal o_SPI_LD_DAC_A : std_logic;
  		signal o_add_board_dacs : std_logic_vector(7 downto 0) := (others => '0');
  		signal wadd_board : std_logic_vector(15 downto 0);
  		signal wdata_board : std_logic_vector(31 downto 0);
  		signal i_data_board_dacs : std_logic_vector(31 downto 0);
      signal o_data_board_dacs : std_logic_vector(31 downto 0);
  		signal o_wen_data_board_dacs : std_logic;
  		signal wboard : std_logic;
      signal board_state : std_logic_vector(3 downto 0);
  		
  		constant ckTime: 		time	:= 10 ns;
		
begin
  --  Component instantiation.
  
  reset <= not reset_n;
  
  -- generate the clock
ckProc: process
begin
   clk <= '0';
   wait for ckTime/2;
   clk <= '1';
   wait for ckTime/2;
end process;

inita : process
begin
   reset_n	 <= '0';
   wait for 8 ns;
   reset_n	 <= '1';
   
   wait;
end process inita;

board_ram : component ip_ram
    Port map ( 
    clock       => clk,
    wren        => wboard,
    wraddress   => wadd_board(7 downto 0),
    rdaddress   => o_add_board_dacs,
    data        => wdata_board,
    q           => i_data_board_dacs--,
);

board_ram_o : component ip_ram
    Port map ( 
    clock       => clk,
    wren        => o_wen_data_board_dacs,
    wraddress   => o_add_board_dacs,
    rdaddress   => o_add_board_dacs,
    data        => o_data_board_dacs,
    q           => open--,
);

chip_ram : component ip_ram
    Port map ( 
    clock       => clk,
    wren        => wchip,
    wraddress   => wadd_chip(7 downto 0),
    rdaddress   => o_add_chip_dacs(7 downto 0),
    data        => wdata_chip,
    q           => i_data_chip_dacs--,
);


-- write board ram
process(clk, reset_n) 
begin
    if(reset_n = '0') then
    wadd_board      <= (others => '1');
    wboard          <= '0'; 
    wdata_board     <= (others => '0');
    board_state     <= (others => '0');
    elsif(rising_edge(clk)) then
    wboard          <= '0';
    wadd_board      <= wadd_board + '1';
    wdata_board     <= (others => '0');
        
    if(board_state = x"0") then
        wboard      <= '1';
        wdata_board <= x"00000001";
        board_state <= x"1";
    end if;
    
    if(board_state = x"1") then
        wboard      <= '1';
        wdata_board <= x"AAAAAAAA";
        board_state <= x"2";
    end if;
    
    if(board_state = x"2") then
        wboard      <= '1';
        wdata_board <= x"BBBBBBBB";
        board_state <= x"3";
    end if;        
        
end if;
end process;


-- write mupix ram
process(clk, reset_n) 
begin
    if(reset_n = '0') then
    wadd_chip      <= (others => '1');
    wchip          <= '0'; 
    wdata_chip     <= (others => '0');
    chip_state     <= (others => '0');
    elsif(rising_edge(clk)) then
    wchip          <= '0';
    wadd_chip      <= wadd_chip + '1';
    wdata_chip    <= (others => '0');
        
    if(chip_state = x"0") then
        wchip      <= '1';
        wdata_chip <= x"00000001";
        chip_state <= x"1";
    end if;
    
    if(chip_state = x"1") then
        wchip      <= '1';
        wdata_chip <= x"AAAAAAAA";
        chip_state <= x"2";
    end if;
    
    if(chip_state = x"2") then
        wchip      <= '1';
        wdata_chip <= x"BBBBBBBB";
        chip_state <= x"3";
    end if;        
        
end if;
end process;


e_mupix_block : component mupix_block
generic map(NCHIPS => 1)
port map (

    -- chip dacs
    i_CTRL_SDO_A            => i_CTRL_SDO_A,
    o_CTRL_SDI_A            => o_CTRL_SDI_A,
    o_CTRL_SCK1_A           => o_CTRL_SCK1_A,
    o_CTRL_SCK2_A           => o_CTRL_SCK2_A,
    o_CTRL_Load_A           => o_CTRL_Load_A,
    o_CTRL_RB_A             => o_CTRL_RB_A,
    i_data_chip_dacs        => i_data_chip_dacs,
    o_add_chip_dacs         => o_add_chip_dacs,
    
    -- board dacs
    i_SPI_DOUT_ADC_0_A      => i_SPI_DOUT_ADC_0_A,
    o_SPI_DIN0_A            => o_SPI_DIN0_A,
    o_SPI_CLK_A             => o_SPI_CLK_A,
    o_SPI_LD_ADC_A          => o_SPI_LD_ADC_A,
    o_SPI_LD_TEMP_DAC_A     => o_SPI_LD_TEMP_DAC_A,
    o_SPI_LD_DAC_A          => o_SPI_LD_DAC_A,
    o_add_board_dacs        => o_add_board_dacs(3 downto 0),
    i_data_board_dacs       => i_data_board_dacs,
    o_data_board_dacs       => o_data_board_dacs,
    o_wen_data_board_dacs   => o_wen_data_board_dacs,
    
    i_ckdiv                 => (others => '0'),

    i_reset                 => reset,
    -- 156.25 MHz
    i_clk                   => clk,
    i_clk125                => clk--,
);


end behav;
