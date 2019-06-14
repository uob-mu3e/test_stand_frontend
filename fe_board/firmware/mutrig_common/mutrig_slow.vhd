-- Slow Controle MuTRiG 
-- Simon Corrodi, May 2017
-- corrodis@phys.ethz.ch

library IEEE;
use IEEE.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use IEEE.numeric_std.all;

entity mutrig_slow is
	generic(
		C_NUM_SPI_CLIENTS	: integer := 1; 
		C_HANDLEID_WIDTH	: integer := 1;
		C_SPI_NUM_WORDS		: integer; 			-- THE TOTAL NUMBER OF WORDS TO READ FROM THE MEMORY
		C_SPI_FIRST_MSB		: integer			-- THE MSB OF THE FIRST WORD TO WRITE
	);
	Port (
		i_clk		: in  std_logic; -- (100 MHz)
		i_rst		: in  std_logic;
		i_start_spi	: in  std_logic;
		i_start_i2c	: in  std_logic; -- config via i2c of si3544
		o_done		: out std_logic;
		o_timeout	: out std_logic;
		i_handleid	: in  std_logic_vector(C_HANDLEID_WIDTH-1 downto 0);
		o_data		: out std_logic_vector(31 downto 0);
		i_data		: in  std_logic_vector(31 downto 0);
		o_wraddr		: out std_logic_vector(15 downto 0);
		o_rdaddr		: out std_logic_vector(15 downto 0);
		o_wren		: out std_logic;
		-- spi signals
		o_mdo		: out std_logic;                                    -- common output to all connected spi clients
		o_sclk		: out std_logic;                                    -- common clock  to all connected spi clients
		i_mdi		: in  std_logic_vector(C_NUM_SPI_CLIENTS-1 downto 0);	-- individual spi slave input
		o_cs		: out std_logic_vector(C_NUM_SPI_CLIENTS-1 downto 0);   -- individual chip select signals

		-- i2c signals
		io_scl		: inout  std_logic;
		io_sda		: inout  std_logic
	);
end mutrig_slow;

architecture RTL of mutrig_slow is

component i2c_control is
	generic(
		C_NUM_WORDS      : integer := 1		                   -- THE TOTAL NUMBER OF BYTES TO READ FROM THE MEMORYRY
	);
	port(
		i_clk         : in  std_logic;                             -- system clock signal (100 MHz?)
		i_rst         : in  std_logic;                             -- reset
		-- control signals from the daq controle
		i_start_trans : in  std_logic;                             -- start transmission from daq controle
		o_eot         : out std_logic;                             -- end of transmission for daq controle
		-- data from or to the memory
		o_read_addr	  : out std_logic_vector(15 downto 0);	   -- memory read address
		i_conf_data	  : in  std_logic_vector(31 downto 0);	   -- configuration data from memory

		o_return_data : out std_logic_vector(31 downto 0);	   -- readback data (to be written to memory)
		o_write_addr  : out std_logic_vector(15 downto 0);	   -- memory write address for returned data
		o_write_en    : out std_logic;                             -- memory write enable 

		-- i2c signals
		io_scl		: inout  std_logic;
		io_sda		: inout  std_logic
	);
end component i2c_control;


component spi_control is
	generic(
		C_NUM_WORDS : integer := 146;				-- THE TOTAL NUMBER OF WORDS TO READ FROM THE MEMORY
		C_FIRST_MSB : integer := 16;				-- THE MSB OF THE FIRST WORD TO WRITE
		C_NUM_CLIENTS	: integer := 1;			-- NUMBER OF CLIENTS CONNECTED TO THE SPI INTERFACE
		--C_CLK_DIVIDE : integer := 10;			-- CLOCK DIVISION FACTOR FOR THE SYSTEM TO SPI CLOCK DIVISION
		C_HANDLEID_WIDTH : integer := 1			-- WIDTH OF THE HANDLEID FOR MULTIPLEXING OF THE SIGNALS
	);
	port(
		i_clk         : in  std_logic;                                    -- system clock signal
		i_rst         : in  std_logic;                                    -- reset
		-- control signals from the daq controle
		i_handleid    : in  std_logic_vector(C_HANDLEID_WIDTH-1 downto 0);-- handle id of chip
		i_start_trans : in  std_logic;                                    -- start transmission from daq controle
		o_eot         : out std_logic;                                    -- end of transmission for daq controle
		-- data from or to the memory
		o_read_addr	  : out std_logic_vector(15 downto 0);	               -- memory read address
		i_conf_data	  : in  std_logic_vector(31 downto 0);	               -- configuration data from memory
		o_return_data : out std_logic_vector(31 downto 0);	               -- readback data (to be written to memory)
		o_write_addr  : out std_logic_vector(15 downto 0);	               -- memory write address for returned data
		o_write_en    : out std_logic;                                    -- memory write enable 
		-- spi signals
		o_mdo         : out std_logic;                                    -- common output to all connected spi clients
		o_sclk        : out std_logic;                                    -- common clock  to all connected spi clients
		i_mdi         : in  std_logic_vector(C_NUM_CLIENTS-1 downto 0);	-- individual spi slave input
		o_cs          : out std_logic_vector(C_NUM_CLIENTS-1 downto 0)	   -- individual chip select signals
	);
end component;

type state_type is (idle_s, config_s, config_clk_s, eot_s, timeout_s);
signal state : state_type;

signal config_d     : std_logic;

signal spi_start_config, spi_eot : std_logic;
signal i2c_start_config, i2c_eot : std_logic;

signal timeout_cnt : std_logic_vector(23 downto 0);
signal timedout : std_logic;    
signal timedout_l : std_logic;    

signal spi_data_out, i2c_data_out : std_logic_vector(31 downto 0);
signal spi_add_out,  i2c_add_out  : std_logic_vector(15 downto 0);
signal spi_add_in,   i2c_add_in   : std_logic_vector(15 downto 0);
signal spi_out_wren, i2c_out_wren : std_logic;

-- 
signal scl_pad_i     : std_logic;                                    -- i2c clock line input
signal scl_pad_o     : std_logic;                                    -- i2c clock line output
signal scl_padoen_o  : std_logic;                                    -- i2c clock line output enable, active low
signal sda_pad_i     : std_logic;                                    -- i2c data line input
signal sda_pad_o     : std_logic;                                    -- i2c data line output
signal sda_padoen_o  : std_logic;                                    -- i2c data line output enable, active low

begin 

 -- direct to spi
u_spi_control : spi_control
	generic map(
		C_NUM_WORDS => C_SPI_NUM_WORDS,
		C_FIRST_MSB => C_SPI_FIRST_MSB,
		C_NUM_CLIENTS=> C_NUM_SPI_CLIENTS,
		C_HANDLEID_WIDTH => C_HANDLEID_WIDTH
	)
	port map(
		i_clk         => i_clk,
		i_rst         => i_rst,
		i_handleid    => i_handleid,
		i_start_trans => spi_start_config,
		o_eot         => spi_eot,
		o_read_addr   => spi_add_in,
		i_conf_data   => i_data,
		o_return_data => spi_data_out,
		o_write_addr  => spi_add_out,
		o_write_en    => spi_out_wren,
		o_mdo         => o_mdo,
		o_sclk        => o_sclk,
		i_mdi         => i_mdi,
		o_cs          => o_cs
	);
 

u_i2c_control : i2c_control
	generic map(
		C_NUM_WORDS => 1
	)
	port map(
		i_clk         => i_clk,
		i_rst         => i_rst or timedout,
		--i_handleid    => i_handleid,
		i_start_trans => i2c_start_config,
		o_eot         => i2c_eot,
		o_read_addr   => i2c_add_in,
		i_conf_data   => i_data,
		o_return_data => i2c_data_out,
		o_write_addr  => i2c_add_out,
		o_write_en    => i2c_out_wren,
		io_scl        => io_scl,
		io_sda        => io_sda
	);

process(i_clk, i_rst)
begin
	if(i_rst = '1') then
		state 		   <= idle_s;
		spi_start_config  <= '0';
		i2c_start_config  <= '0';
		config_d          <= '0';
		timedout          <= '0';
		timedout_l        <= '0';
	elsif(i_clk'event and i_clk = '1') then
		config_d     <=  '0';
		timedout     <= '0';
		case state is
			when idle_s =>
				config_d     <=  '0'; -- explicit, not needed
				timedout     <=  '0';
				if(i_start_spi = '1') then
					state         <= config_s;
				end if;

				if(i_start_i2c  = '1') then
					state         <= config_clk_s;
					timeout_cnt   <= x"01ffff";
				end if;

			when config_s =>
				spi_start_config <= '1';
				if(spi_eot = '1') then
					state            <= eot_s;
					spi_start_config <= '0';
					config_d         <= '1';--(others => '1');
				end if;

			when eot_s =>
				spi_start_config <= '0'; -- not needed
				i2c_start_config <= '0'; -- not needed
				config_d         <= '1';
				if(i_start_spi = '0' and i_start_i2c = '0') then
					state            <= idle_s; -- reset config_done and re-arm
				end if;

			when config_clk_s =>
				i2c_start_config <= '1';
				timedout_l       <= '0';
				if(i2c_eot = '1') then
					state            <= eot_s;
					i2c_start_config <= '0';
					config_d         <= '1';--(others => '1');
				end if;
				if(timeout_cnt = x"000000") then
					state            <= timeout_s;
				else
					timeout_cnt <= std_logic_vector(unsigned(timeout_cnt)-1);
				end if;

			when timeout_s =>
				spi_start_config <= '0';
				timedout_l       <= '1';
				timedout         <= '1';
				config_d         <= '1'; --(others => '1');
				state            <= eot_s;
		end case;
	end if;
end process;
				  
  
with state select o_rdaddr <=
	spi_add_in when config_s,
	i2c_add_in when config_clk_s,
	spi_add_in when others;

with state select o_data <=
	spi_data_out when config_s,
	i2c_data_out when config_clk_s,
	spi_data_out when others;

with state select o_wraddr <=
	spi_add_out when config_s,
	i2c_add_out when config_clk_s,
	spi_add_out when others;

with state select o_wren <=
	spi_out_wren when config_s,
	i2c_out_wren when config_clk_s,
	spi_out_wren when others;
				  
o_done   <= config_d;
o_timeout <= timedout_l;
		
end RTL;
