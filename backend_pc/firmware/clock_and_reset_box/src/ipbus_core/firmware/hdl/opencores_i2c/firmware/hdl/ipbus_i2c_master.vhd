-- ipbus_i2c_master
--
-- Wrapper for opencores i2c wishbone slave
--
-- Dave Newbold, Jan 2012

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;
use work.ipbus.all;

entity ipbus_i2c_master is
	generic(addr_width: natural := 0);
	port(
		clk: in std_logic;
		rst: in std_logic;
		ipbus_in: in ipb_wbus;
		ipbus_out: out ipb_rbus;
		ipbus_in_fast: in ipb_wbus;
        ipbus_out_fast: out ipb_rbus;
 		ipbus_in_mem: in ipb_wbus;
        ipbus_out_mem: out ipb_rbus;       
		scl_o: out std_logic;
		scl_i: in std_logic;
		sda_o: out std_logic;
		sda_i: in std_logic
	);
	
end ipbus_i2c_master;

architecture rtl of ipbus_i2c_master is

	signal stb, stb_x, ack, ack_last, sda_enb: std_logic;
	
	signal addr:       std_logic_vector(2 downto 0);
	signal data_in:    std_logic_vector(7 downto 0);
	signal data_out:   std_logic_vector(7 downto 0);
	signal we:         std_logic;
	signal sr:         std_logic_vector(7 downto 0);
	
	type state_type is (idle, simple, write1, write2, write3, write4, write5, write6, write7, read1, read2, read3, read4);
	signal state : state_type;
	signal statenext : state_type;
	
	
	
	constant ADDR_I2C_CTRL: std_logic_vector(2 downto 0) := "010";
	constant ADDR_I2C_DATA: std_logic_vector(2 downto 0) := "011";
	constant ADDR_I2C_CMD_STAT: std_logic_vector(2 downto 0) := "100";
	
	constant I2C_START: std_logic_vector(7 downto 0) := X"90";
	constant I2C_READ: std_logic_vector(7 downto 0) := X"20";
	constant I2C_READPLUSNACK: std_logic_vector(7 downto 0) := X"28";
	constant I2C_STOP: std_logic_vector(7 downto 0) := X"40";
	constant I2C_WRITE: std_logic_vector(7 downto 0) := X"10";

    constant DAUGHTERBIT: integer := 1;
    
    subtype addr_t is std_logic_vector(7 downto 0);
    subtype daughterdevaddr_t is std_logic_vector(6 downto 0);
    type daughteraddrs is array (7 downto 0) of daughterdevaddr_t;
    constant DAUGHTERDEVADDRS : daughteraddrs := ("1110000","1110100","1110001","1110101","1110010","1110110","1110011","1110111");
    subtype DEVRANGE is natural range 30 downto 28;

    constant TIPBIT : integer := 1;
    constant BUSYBIT : integer := 6;
    constant NOACKBIT:    integer := 7;
    
    signal devaddr : addr_t;
    signal busaddr : addr_t;
    signal ackseen: std_logic;
    
    signal bytecount : natural range 4 downto 0;

begin

    stb <= stb_x and not ack;

    process(clk, rst)
    begin
    if(rst = '1') then
        state <= idle;
        stb_x  <= '0';
        we   <= '0';
        ack_last    <= '0';
        ackseen     <= '0';
    elsif(rising_edge(clk)) then
        ack_last <= ack;
        stb_x  <= '0';
        case state is
            when idle =>
                if(ipbus_in.ipb_strobe = '1')then
                    state   <=  simple;
                    addr    <=  ipbus_in.ipb_addr(2 downto 0);
                    data_in <=  ipbus_in.ipb_wdata(7 downto 0);
                    we      <=  ipbus_in.ipb_write;
                    stb_x   <= '1';
                elsif(ipbus_in_fast.ipb_strobe = '1') then 
                    state   <= write1;
                    addr    <= ADDR_I2C_DATA;
                    data_in <= ipbus_in.ipb_addr(31 downto 25) & "0";
                    we      <= '1';
                    stb_x   <= '1';
                   ipbus_out.ipb_rdata(31 downto 0) <= (others => '0');
                end if;
            when simple =>
                ipbus_out.ipb_rdata(7 downto 0) <= data_out;
                ipbus_out.ipb_rdata(31 downto 8) <= (others => '0');
                ipbus_out.ipb_err <= '0';
                ipbus_out.ipb_ack <= ack;
                if(ack = '1') then
                    state <= idle;
                    we    <= '0';
                 end if;
            when write1 =>
                if(ack = '1') then
                    state <= write2;
                    addr    <= ADDR_I2C_CMD_STAT;
                    data_in <= I2C_WRITE;
                    we      <= '1';
                    stb_x   <= '1';
                    ackseen <= '0';
                end if;
            when write2 =>
                if(ack = '1') then
                    ackseen <= '1';
                    stb_x   <= '0';
                end if;
                if(sr(TIPBIT) = '0' and (ack = '1' or ackseen = '1')) then
                    if(ipbus_in.ipb_wdata(24) = '1') then
                        state   <= write3;
                        addr    <= ADDR_I2C_DATA;
                        data_in <= ipbus_in.ipb_addr(23 downto 16);
                        we      <= '1';
                        stb_x   <= '1';
                    else 
                        if( ipbus_in.ipb_write = '1') then
                            state   <= write5;
                            addr    <= ADDR_I2C_DATA;
                            data_in <= ipbus_in.ipb_wdata(7 downto 0);
                            we      <= '1';
                            stb_x   <= '1';
                         else
                            state   <= read1;
                            addr    <= ADDR_I2C_DATA;
                            data_in <= ipbus_in.ipb_addr(31 downto 25) & "0";
                            we      <= '1';
                            stb_x   <= '1';
                            bytecount   <= conv_integer(ipbus_in.ipb_addr(15 downto 14)) + 1;
                         end if;
                    end if;
                    ackseen <= '0'; 
                    if(sr(NOACKBIT) = '1') then
                        ipbus_out.ipb_rdata(7 downto 0)  <= (others => '0');
                        ipbus_out.ipb_rdata(31 downto 8) <= (others => '1');
                        ipbus_out.ipb_err <= '0';
                        ipbus_out.ipb_ack <= ack;
                        state   <= idle;
                    end if;
                end if;
             when write3 =>
                if(ack = '1') then
                    state <= write4;
                    addr    <= ADDR_I2C_CMD_STAT;
                    data_in <= I2C_WRITE;
                    we      <= '1';
                    stb_x   <= '1';
                    ackseen <= '0';
                end if;
             when write4 =>
                if(ack = '1') then
                    ackseen <= '1';
                    stb_x   <= '0';
                end if;
                if(sr(TIPBIT) = '0' and (ack = '1' or ackseen = '1')) then
                    state   <= write5;
                    addr    <= ADDR_I2C_DATA;
                    data_in <= ipbus_in.ipb_wdata(7 downto 0);
                    we      <= '1';
                    stb_x   <= '1';
                end if;
             when write5 =>
                 if(ack = '1') then
                    state <= write6;
                    addr    <= ADDR_I2C_CMD_STAT;
                    data_in <= I2C_WRITE;
                    we      <= '1';
                    stb_x   <= '1';
                    ackseen <= '0';
                end if;
            when write6 =>
                if(ack = '1') then
                    ackseen <= '1';
                    stb_x   <= '0';
                end if;
                if(sr(TIPBIT) = '0' and (ack = '1' or ackseen = '1')) then
                    state <= write7;
                    addr    <= ADDR_I2C_CMD_STAT;
                    data_in <= I2C_STOP;
                    we      <= '1';
                    stb_x   <= '1';
                    ackseen <= '0';
                end if;
            when write7 =>
                if(ack = '1') then
                    ackseen <= '1';
                    stb_x   <= '0';
                end if;
                if(sr(BUSYBIT) = '0' and (ack = '1' or ackseen = '1')) then
                    state   <= idle;
                    we      <= '0';
                    stb_x   <= '0';
                    ackseen <= '0';
                    ipbus_out.ipb_err <= '0';
                    ipbus_out.ipb_ack <= ack;                    
                end if;
            when read1 =>
                if(ack = '1') then
                    state <=read2;
                    addr    <= ADDR_I2C_CMD_STAT;
                    data_in <= I2C_WRITE;
                    we      <= '1';
                    stb_x   <= '1';
                    ackseen <= '0';
                end if;
            when read2 =>          
                if(ack = '1') then
                    ackseen <= '1';
                    stb_x   <= '0';
                end if;
                if(sr(TIPBIT) = '0' and (ack = '1' or ackseen = '1')) then
                    state   <= read3;
                    addr    <= ADDR_I2C_CMD_STAT;
                    if(bytecount = 1) then
                        data_in <= I2C_READPLUSNACK;
                    else
                        data_in <= I2C_READ;
                    end if;
                    bytecount   <= bytecount -1;
                    we      <= '1';
                    stb_x   <= '1';
                end if;
           when read3 =>
                if(ack = '1') then
                    ackseen <= '1';
                    stb_x   <= '0';
                end if;
                if(sr(TIPBIT) = '0' and (ack = '1' or ackseen = '1')) then
                    state   <= read4;
                    addr    <= ADDR_I2C_DATA;
                    we      <= '0';
                    stb_x   <= '1';
                end if;
           when read4 =>
                if(ack = '1') then
                    ipbus_out.ipb_rdata(8*bytecount+7 downto 8*bytecount)  <= data_out;
                    if(bytecount = 0) then
                        state   <= write7;
                        addr    <= ADDR_I2C_CMD_STAT;
                        data_in <= I2C_STOP;
                        we      <= '1';
                        stb_x   <= '1';
                        ackseen <= '0';
                    else
                        state   <= read3;
                        addr    <= ADDR_I2C_CMD_STAT;
                        if(bytecount = 1) then
                            data_in <= I2C_READPLUSNACK;
                        else
                            data_in <= I2C_READ;
                        end if;
                        bytecount   <= bytecount -1;
                        we      <= '1';
                        stb_x   <= '1';
                    end if;
                end if;
        end case;
    end if;
    end process;


	--stb <= ipbus_in.ipb_strobe and not ack;

	i2c: entity work.i2c_master_top port map(
		wb_clk_i => clk,
		wb_rst_i => rst,
		arst_i => '1',
		wb_adr_i => addr,
		wb_dat_i => data_in,
		wb_dat_o => data_out,
		wb_we_i => we,
		wb_stb_i => stb,
		wb_cyc_i => '1',
		wb_ack_o => ack,
		scl_pad_i => scl_i,
		scl_padoen_o => scl_o,
		sda_pad_i => sda_i,
		sda_padoen_o => sda_o,
		sr           => sr
	);
		
end rtl;
