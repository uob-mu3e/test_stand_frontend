library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
--use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
use std.textio.all;
use IEEE.std_logic_textio.all; 


--  A testbench has no ports.
entity mupix_block_tb is
end entity;

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
      
      
      -- board dacs
      i_SPI_DOUT_ADC_0_A      : in std_logic;
      o_SPI_DIN0_A            : out std_logic;
      o_SPI_CLK_A             : out std_logic;
      o_SPI_LD_ADC_A          : out std_logic;
      o_SPI_LD_TEMP_DAC_A     : out std_logic;
      o_SPI_LD_DAC_A          : out std_logic;
      
      

      -- mupix dac regs
      i_reg_add               : in std_logic_vector(7 downto 0);
      i_reg_re                : in std_logic;
      o_reg_rdata             : out std_logic_vector(31 downto 0);
      i_reg_we                : in std_logic;
      i_reg_wdata             : in std_logic_vector(31 downto 0);
      
      i_ckdiv                 : in std_logic_vector(15 downto 0);

      i_reset                 : in std_logic;
      -- 156.25 MHz
      i_clk                   : in std_logic;
      i_clk125                : in std_logic--;
    );
	end component mupix_block;

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
  		
  		signal i_SPI_DOUT_ADC_0_A : std_logic;
  		signal o_SPI_DIN0_A : std_logic;
  		signal o_SPI_CLK_A : std_logic;
  		signal o_SPI_LD_ADC_A : std_logic;
  		signal o_SPI_LD_TEMP_DAC_A : std_logic;
  		signal o_SPI_LD_DAC_A : std_logic;

      signal reg_add : std_logic_vector(7 downto 0);
      signal reg_re : std_logic;
      signal reg_rdata : std_logic_vector(31 downto 0);
      signal reg_we : std_logic;
      signal reg_wdata : std_logic_vector(31 downto 0);

      signal state : std_logic_vector(7 downto 0);
  		
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

  -- test mupix block board dac
  process(clk, reset_n) 
  begin
      if(reset_n = '0') then
        reg_add   <= (others => '0');
        reg_wdata <= (others => '0');
        reg_re    <= '0'; 
        reg_we    <= '0'; 
        state     <= (others => '0');
      elsif(rising_edge(clk)) then
        state <= state + '1';
        reg_re    <= '0';
        reg_we    <= '0';

        if( state = x"0" ) then
            reg_we      <= '1';
            reg_wdata   <= x"00000001";
            reg_add     <= x"81";
        end if;
        
        if(state = x"1") then
            reg_we      <= '1';
            reg_wdata   <= x"00000002";
            reg_add     <= x"81";
        end if;
        
        if(state = x"2") then
            reg_we      <= '1';
            reg_wdata   <= x"AAAAAAAA";
            reg_add     <= x"80";
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
    
    -- board dacs
    i_SPI_DOUT_ADC_0_A      => i_SPI_DOUT_ADC_0_A,
    o_SPI_DIN0_A            => o_SPI_DIN0_A,
    o_SPI_CLK_A             => o_SPI_CLK_A,
    o_SPI_LD_ADC_A          => o_SPI_LD_ADC_A,
    o_SPI_LD_TEMP_DAC_A     => o_SPI_LD_TEMP_DAC_A,
    o_SPI_LD_DAC_A          => o_SPI_LD_DAC_A,
    
    -- mupix dac regs
    i_reg_add               => reg_add,
    i_reg_re                => reg_re,
    o_reg_rdata             => reg_rdata,
    i_reg_we                => reg_we,
    i_reg_wdata             => reg_wdata,
    
    i_ckdiv                 => (others => '0'),

    i_reset                 => reset,
    -- 156.25 MHz
    i_clk                   => clk,
    i_clk125                => clk--,
);


end architecture;
