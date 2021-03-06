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
	
	signal state_code: std_logic_vector(7 downto 0);
	
	
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
    signal tipseen: std_logic;
    signal strobe_last: std_logic;
    
    
    signal bytecount : natural range 4 downto 0;

begin


   ila1: entity work.ila_0
        port map( 
    clk  => clk,--: in STD_LOGIC;
    probe0(0) => ipbus_in_fast.ipb_strobe,
    probe1(0) => ipbus_in_fast.ipb_write,
    probe2    => ipbus_in_fast.ipb_addr,
    probe3(0) => stb,
    probe4(0) => ack,
    probe5(31 downto 0)  => ipbus_out_fast.ipb_rdata,
    --probe5(31 downto 3)  => (others => '0'),
    probe6(0) => ackseen,
    probe7(0) => sr(TIPBIT),
    probe8(7 downto 0)    => data_in,
    probe8(15 downto 8)   => data_out,
    probe8(18 downto 16)   => addr, 
    probe8(31 downto 19)   => (others => '0'),
    probe9(0) => sr(BUSYBIT),
    probe10(0) => sr(NOACKBIT),
    probe11(7 downto 0)    => state_code,
    probe11(31 downto 8)  => (others => '0'),
    probe12(0) => ipbus_out_fast.ipb_ack,
    probe13(0) => scl_i,
    probe14(0) => tipseen,
    probe15(0) => sda_i 
 );

    state_code <= "00000001" when state = idle else
                  "00000010" when state = simple else
                  "00000100" when state = write1 else
                  "00000101" when state = write2 else
                  "00000110" when state = write3 else
                  "00000111" when state = write4 else
                  "00001000" when state = write5 else
                  "00001001" when state = write6 else
                  "00001011" when state = write7 else
                  "00010000" when state = read1 else
                  "00010001" when state = read2 else 
                  "00010010" when state = read3 else                                        
                  "00010011" when state = read4 else
                  "00000000";


    stb <= stb_x and not ack;

    process(clk, rst)
    begin
    if(rst = '1') then
        state <= idle;
        stb_x  <= '0';
        we   <= '0';
        ack_last    <= '0';
        ackseen     <= '0';
        tipseen     <= '0';
        strobe_last <= '0';
        ipbus_out.ipb_ack <= '0';
        ipbus_out_fast.ipb_ack <= '0';
        ipbus_out_mem.ipb_ack <= '0'; 
        ipbus_out.ipb_err <= '0';
        ipbus_out_fast.ipb_err <= '0';
        ipbus_out_mem.ipb_err <= '0'; 
    elsif(rising_edge(clk)) then
        ack_last <= ack;
        stb_x  <= '0';
        
        ipbus_out.ipb_ack <= '0';
        ipbus_out_fast.ipb_ack <= '0';
        ipbus_out_mem.ipb_ack <= '0'; 
        
        ipbus_out.ipb_err <= '0';
        ipbus_out_fast.ipb_err <= '0';
        ipbus_out_mem.ipb_err <= '0'; 
        
        strobe_last <= ipbus_in_fast.ipb_strobe;
        
        case state is
            when idle =>
                if(ipbus_in.ipb_strobe = '1')then
                    state   <=  simple;
                    addr    <=  ipbus_in.ipb_addr(2 downto 0);
                    data_in <=  ipbus_in.ipb_wdata(7 downto 0);
                    we      <=  ipbus_in.ipb_write;
                    stb_x   <= '1';
                elsif(ipbus_in_fast.ipb_strobe = '1' and strobe_last = '0') then 
                    -- Put the device address in the data register
                    state   <= write1;
                    addr    <= ADDR_I2C_DATA;
                    data_in <= ipbus_in_fast.ipb_addr(31 downto 25) & "0";
                    we      <= '1';
                    stb_x   <= '1';
                   ipbus_out_fast.ipb_rdata(31 downto 0) <= (others => '0');
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
                    -- Start I2C with sending the device address
                    state <= write2;
                    addr    <= ADDR_I2C_CMD_STAT;
                    data_in <= I2C_START;
                    we      <= '1';
                    stb_x   <= '1';
                    ackseen <= '0';
                    tipseen <= '0';
                end if;
            when write2 =>
                if(ack = '1') then
                    ackseen <= '1';
                    stb_x   <= '0';
                end if;
                if(sr(TIPBIT) = '1') then
                    tipseen <= '1';
                end if;
                if(sr(TIPBIT) = '0' and tipseen = '1' and (ack = '1' or ackseen = '1')) then
                    ackseen <= '0';
                    tipseen <= '0';
                    if(ipbus_in_fast.ipb_addr(24) = '1') then
                        -- Put the register address in the data regfister
                        state   <= write3;
                        addr    <= ADDR_I2C_DATA;
                        data_in <= ipbus_in_fast.ipb_addr(23 downto 16);
                        we      <= '1';
                        stb_x   <= '1';
                    else 
                        if( ipbus_in_fast.ipb_write = '1') then
                            -- Put the write data in the data register
                            state   <= write5;
                            addr    <= ADDR_I2C_DATA;
                            data_in <= ipbus_in_fast.ipb_wdata(7 downto 0);
                            we      <= '1';
                            stb_x   <= '1';
                         else
                            state   <= read1;
                            addr    <= ADDR_I2C_DATA;
                            data_in <= ipbus_in_fast.ipb_addr(31 downto 25) & "0";
                            we      <= '1';
                            stb_x   <= '1';
                            bytecount   <= conv_integer(ipbus_in_fast.ipb_addr(15 downto 14)) + 1;
                         end if;
                    end if;
                    ackseen <= '0'; 
                    if(sr(NOACKBIT) = '1') then
                        -- There was no acknowledge from the device (maybe better error handling?)
                        ipbus_out_fast.ipb_rdata(7 downto 0)  <= (others => '0');
                        ipbus_out_fast.ipb_rdata(31 downto 8) <= (others => '1');
                        ipbus_out_fast.ipb_err <= '0';
                        ipbus_out_fast.ipb_ack <= '1';
                        state   <= idle;
                    end if;
                end if;
             when write3 =>
                if(ack = '1') then
                    -- Write the register address
                    state <= write4;
                    addr    <= ADDR_I2C_CMD_STAT;
                    data_in <= I2C_WRITE;
                    we      <= '1';
                    stb_x   <= '1';
                    ackseen <= '0';
                    tipseen <= '0';
                end if;
             when write4 =>
                if(ack = '1') then
                    ackseen <= '1';
                    stb_x   <= '0';
                end if;
                if(sr(TIPBIT) = '1') then
                    tipseen <= '1';
                end if;
                if(sr(TIPBIT) = '0' and tipseen = '1' and (ack = '1' or ackseen = '1')) then
                    if( ipbus_in_fast.ipb_write = '1') then
                        -- write the data to be written
                        state   <= write5;
                        addr    <= ADDR_I2C_DATA;
                        data_in <= ipbus_in_fast.ipb_wdata(7 downto 0);
                        we      <= '1';
                        stb_x   <= '1';
                        ackseen <= '0';
                        tipseen <= '0';
                    else
                        -- set device address again, this time for read
                        state   <= read1;
                        addr    <= ADDR_I2C_DATA;
                        data_in <= ipbus_in_fast.ipb_addr(31 downto 25) & "1";
                        we      <= '1';
                        stb_x   <= '1';
                        bytecount   <= conv_integer(ipbus_in_fast.ipb_addr(15 downto 14)) + 1;
                        ackseen <= '0';
                        tipseen <= '0';
                    end if;
                end if;
             when write5 =>
                 if(ack = '1') then
                    state <= write6;
                    addr    <= ADDR_I2C_CMD_STAT;
                    data_in <= I2C_WRITE;
                    we      <= '1';
                    stb_x   <= '1';
                    ackseen <= '0';
                    tipseen <= '0';
                end if;
            when write6 =>
                if(ack = '1') then
                    ackseen <= '1';
                    stb_x   <= '0';
                end if;
                if(sr(TIPBIT) = '1') then
                    tipseen <= '1';
                end if;
                if(sr(TIPBIT) = '0' and tipseen = '1' and (ack = '1' or ackseen = '1')) then
                    state <= write7;
                    addr    <= ADDR_I2C_CMD_STAT;
                    data_in <= I2C_STOP;
                    we      <= '1';
                    stb_x   <= '1';
                    ackseen <= '0';
                    tipseen <= '0';
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
                    ipbus_out_fast.ipb_err <= '0';
                    ipbus_out_fast.ipb_ack <= '1';                    
                end if;
            when read1 =>
                if(ack = '1') then
                    -- clock out the register address to be read
                    state <=read2;
                    addr    <= ADDR_I2C_CMD_STAT;
                    data_in <= I2C_START;
                    we      <= '1';
                    stb_x   <= '1';
                    ackseen <= '0';
                end if;
            when read2 =>          
                if(ack = '1') then
                    ackseen <= '1';
                    stb_x   <= '0';
                end if;
                if(sr(TIPBIT) = '1') then
                    tipseen <= '1';
                end if;
                if(sr(TIPBIT) = '0' and tipseen = '1' and (ack = '1' or ackseen = '1')) then
                    -- Read from I2C
                    state   <= read3;
                    tipseen <= '0';
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
                if(sr(TIPBIT) = '1') then
                    tipseen <= '1';
                end if;
                if(sr(TIPBIT) = '0' and tipseen = '1' and (ack = '1' or ackseen = '1')) then
                    -- set address to data
                    state   <= read4;
                    addr    <= ADDR_I2C_DATA;
                    we      <= '0';
                    stb_x   <= '1';
                    ackseen <= '0';
                    tipseen <= '0';
                end if;
           when read4 =>
                if(ack = '1') then
                    -- capture data, the stop or read another byte
                    ipbus_out_fast.ipb_rdata(8*bytecount+7 downto 8*bytecount)  <= data_out;
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
