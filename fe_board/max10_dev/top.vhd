library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

entity top is
port (

	CLOCK : in std_logic; -- 50 MHz
	RESET_N : in std_logic;
--
--	Arduino_IO13 : in std_logic;
--	Arduino_IO12 : out std_logic;
--	Arduino_IO11 : out std_logic;
--	Arduino_IO10 : out std_logic;
    
    MSCB_IN : in std_logic;
    MSCB_OUT :out std_logic;
    MSCB_OE : out std_logic;
   -- DIFFIO_B16N : out std_logic;
    --DIFFIO_B14P : out std_logic--;
    
    --Arduino_A0  : in std_logic;
----
	SWITCH1 : in std_logic;
----	SWITCH2 : in std_logic;
----	SWITCH3 : in std_logic;
----	SWITCH4 : in std_logic;
----    SWITCH5 : in std_logic;
--
--
	LED1 : out std_logic;
	LED2 : out std_logic;
	LED3 : out std_logic;
	LED4 : out std_logic;
	LED5 : out std_logic--;

);
end entity;

architecture arch of top is

--    component my_altmult_add IS
--	PORT
--	(
--		clock0		: IN STD_LOGIC  := '1';
--		dataa_0		: IN STD_LOGIC_VECTOR (11 DOWNTO 0) :=  (OTHERS => '0');
--		datab_0		: IN STD_LOGIC_VECTOR (11 DOWNTO 0) :=  (OTHERS => '0');
--		result		: OUT STD_LOGIC_VECTOR (63 DOWNTO 0)
--	);
--    END component my_altmult_add;

    component mscb IS

    port (
        nios_clk                    : in    std_logic;
        reset                       : in    std_logic;
        mscb_to_nios_parallel_in    : out   std_logic_vector(11 downto 0);
        mscb_from_nios_parallel_out : in    std_logic_vector(11 downto 0);
        mscb_data_in                : in    std_logic;
        mscb_data_out               : out   std_logic;
        mscb_oe                     : out   std_logic;
        mscb_counter_in             : out   unsigned(15 downto 0);
        o_mscb_irq                  : out   std_logic;
        i_mscb_address              : in    std_logic_vector(15 downto 0)--;
    );
    END component mscb;

    constant count_max : unsigned(24 downto 0) := (others => '1');
    constant address :   std_logic_vector(15 downto 0) := x"ABCD" ; --Address of board
    
	signal adc_clk : std_logic; -- 10 MHz
	signal nios_clk : std_logic; -- 50 MHz

	signal pll_locked, slow_clk : std_logic;
    signal adc_response_valid ,adc_response_valid_save: std_logic;

	signal sw : std_logic_vector(4 downto 0);
	signal led : std_logic_vector(4 downto 0);

	signal nios_pio : std_logic_vector(31 downto 0);
    
    signal adc_data : std_logic_vector(11 downto 0);
    --signal adc_data_store : std_logic_vector(11 downto 0);

    signal x2 : STD_LOGIC_VECTOR(63 DOWNTO 0);
    
    signal e_x,e_x2 : unsigned(63 downto 0);
    
    signal counter : unsigned(24 downto 0);
    signal mult_delay_cnt : unsigned(10 downto 0);
    
    signal mscb_out_parallel, mscb_in_parallel : std_logic_vector(11 downto 0);
    signal crc_check_en: std_logic;
    signal address_enable : std_logic_vector(1 downto 0); --one hot encoding
   
   
    TYPE state_type IS (idle, addr_cmd_msb, addr_cmd_lsb, crc_check, command_processing, full_operation, CMD_READ_MEM, CMD_READ_MEM_ANSWER);  -- Define the states
    SIGNAL state : state_type;    -- Create a signal that
    signal msg_save: std_logic_vector(31 downto 0);
    signal crc_cnt : unsigned(6 downto 0) := (others => '0');
    signal read_receive_cnt,read_mem_answer_cnt : unsigned(3 downto 0) := (others => '0');
    signal  adc_cnt : unsigned(3 downto 0) := (others => '0');
    signal crc_reg: std_logic_vector(8 downto 0);
    
    
    signal receive_crc, numb_data_bytes, receive_ch_address: std_logic_vector(7 downto 0);
    
    type array5 is array (integer range <>) of std_logic_vector(4 downto 0);
    type array8 is array (integer range <>) of std_logic_vector(7 downto 0);
    type array12 is array (integer range <>) of std_logic_vector(11 downto 0);
    signal full_operation_save: array8(8 downto 0);
    signal full_cmd_save, testfifo: std_logic_vector(7 downto 0);
    
    constant cmd_read_mem_adc : array8(6 downto 0) := (x"07"       ,x"00",x"02",x"00",x"12",x"34",x"00");
    signal cmd_read_mem_answ : array8(5 downto 0) := ("01111111", x"80", x"02",x"12", x"34", x"36");
    signal cmd_read_mem_answ_total: array8(21 downto 0) := ("01111111", x"80", x"02",x"00", x"00",x"00", x"00",x"00", x"00",x"00", x"00",x"00", x"00",x"00", x"00",x"00", x"00",x"00", x"00",x"00", x"00", x"36");
    
    
    signal adc_data_store : array12(8 downto 0) := (others => (others => '0'));
    
    constant adc_channels : array5(8 downto 0) := ( "00010","00001","00000","00111","00110","00101","00100", "00011","10001");
    
    signal response_channel : std_logic_vector(4 downto 0);
    
begin


    my_mscb: mscb
    port map(
        nios_clk                    =>  CLOCK,
        reset                       => not RESET_N,
        mscb_to_nios_parallel_in    => mscb_in_parallel,
        mscb_from_nios_parallel_out => mscb_out_parallel,
        mscb_data_in                => MSCB_IN,--: in    std_logic;
        mscb_data_out               => MSCB_OUT,--: out   std_logic;
        mscb_oe                     => MSCB_OE,--: out   std_logic;
        mscb_counter_in             => open,
        o_mscb_irq                  => open,
        i_mscb_address              => address
    );

    
    
    process(CLOCK,reset_n)
    begin
        if reset_n = '0' then
            mscb_out_parallel <= (others => '0');
            msg_save <= (others => '0');
            crc_check_en <= '0';
            crc_cnt <= (others => '0');
            adc_cnt <= (others => '0');
            adc_response_valid_save <= '0'; 
            adc_data_store <= (others => (others => '0'));
            
        elsif rising_edge(CLOCK) then
            adc_response_valid_save <= adc_response_valid;
            if adc_response_valid ='1' and adc_response_valid /= adc_response_valid_save and response_channel = adc_channels(to_integer(adc_cnt))then
                adc_data_store(to_integer(adc_cnt)) <= adc_data;
                adc_cnt <= adc_cnt +1;
                if adc_cnt = 8 then
                    adc_cnt <=  (others => '0');
                end if;
            end if;

            
            case state is
                when idle =>
                    mscb_out_parallel <= (others => '0');
                    msg_save <= (others => '0');
                    full_cmd_save <= (others => '0');
                    full_operation_save <= (others => x"00");
                    read_receive_cnt <= (others => '0') ;
                    read_mem_answer_cnt <= (others => '0');
                    
                    if mscb_in_parallel(9) = '0' then  --fifo not empty
                        if mscb_in_parallel(8) = '1' then -- address bit set
                            state <= addr_cmd_msb; 
                            msg_save(31 downto 24) <= mscb_in_parallel(7 downto 0);
                            mscb_out_parallel(10) <= '1';  -- read fifo
                        else
                            mscb_out_parallel(10) <= not mscb_out_parallel(10);
                        end if;
                    end if;
                    
                when addr_cmd_msb =>
                    if mscb_in_parallel(7 downto 0) /= msg_save(31 downto 24) then  --wait until fifo changed (failes for msb address = CMD_PING16 or CMD_ADDR_NODE16, ...)
                        if mscb_in_parallel(7 downto 0) = address(15 downto 8) then  --check msb address
                            state <= addr_cmd_lsb;
                            msg_save(23 downto 16) <= mscb_in_parallel(7 downto 0);
                            mscb_out_parallel(10) <= not mscb_out_parallel(10); 
                        else
                            state <= idle;
                        end if;

                    elsif mscb_in_parallel(9) = '0' then
                        mscb_out_parallel(10) <= not mscb_out_parallel(10);
                    end if;
                    
                    
                when addr_cmd_lsb =>
                    if mscb_in_parallel(7 downto 0) /= msg_save(23 downto 16) then --fails if msb = lsb of address
                        if mscb_in_parallel(7 downto 0) = address(7 downto 0) then --check lsb address
                            state <= crc_check;
                            msg_save(15 downto 8) <= mscb_in_parallel(7 downto 0);
                            mscb_out_parallel(10) <= not mscb_out_parallel(10);
                        else
                            state <= idle;
                        end if;

                    elsif mscb_in_parallel(9) = '0' then
                        mscb_out_parallel(10) <= not mscb_out_parallel(10);
                    end if;
                    
                when crc_check =>
--                    if crc_check_en = '1' then
--                        --state <= command_processing;
--                        if crc_cnt <= 32 then
--                            --x8+x5+x4+1
--                            msg_save(31 downto 30) <= msg_save(30 downto 29);
--                            msg_save(29) <= msg_save(31) xor msg_save(28);
--                            msg_save(28) <= msg_save(31) xor msg_save(27);
--                            msg_save(27 downto 25) <= msg_save(26 downto 24);
--                            msg_save(24) <= msg_save(31) xor msg_save(23);
--                            msg_save(23 downto 1) <= msg_save(22 downto 0);
--                            msg_save(0) <= '0';
--                            crc_cnt <= crc_cnt +1;
--                            
--                        else
--                            if msg_save(31 downto 24) = "00000000" then
--                                state <= command_processing;
--                            else
--                                state <= idle;
--                            end if;
--                        end if; 
--                    els
                      if mscb_in_parallel(7 downto 0) /= msg_save(15 downto 8) then  -- fails if lsb address = crc
                        --crc_check_en <= '1';
                        msg_save(7 downto 0) <= mscb_in_parallel(7 downto 0);
                        mscb_out_parallel(10) <= not mscb_out_parallel(10);
                        state <= command_processing;
                    elsif mscb_in_parallel(9) = '0' then
                        mscb_out_parallel(10) <= not mscb_out_parallel(10);
                    end if;
                    
                when command_processing =>
                    if msg_save(31 downto 24) = x"1A" then --ping
                        mscb_out_parallel(10 downto 0) <= "01001111000";
                        state <= idle;
                    elsif msg_save(31 downto 24) = x"0A" then --address
                        state <= full_operation;
                        mscb_out_parallel(10) <= '0';
                    elsif mscb_in_parallel(9) = '0' then
                        mscb_out_parallel(10) <= not mscb_out_parallel(10);
                    end if;
                
                when full_operation =>
                    if full_cmd_save = x"BF"  then --CMD_READ_MEM
                        state <= CMD_READ_MEM;
                        read_receive_cnt <= (others => '0');
                        mscb_out_parallel(10) <= '0';
                    end if;
                    -- return to idle if not
                    if mscb_in_parallel(9) = '0' then  --fifo not empty
                        full_cmd_save <= mscb_in_parallel(7 downto 0);
                        mscb_out_parallel(10) <= not mscb_out_parallel(10);
                    end if;
                    
                when CMD_READ_MEM =>  
                    if mscb_in_parallel(9) = '0' then
                        mscb_out_parallel(10) <= '1';
                    end if;
                    
                    if read_receive_cnt = 9 then
                        -- write adc in memanswer array
                        if receive_ch_address /= x"FF" then
                            cmd_read_mem_answ(1) <= adc_data_store(to_integer(unsigned(receive_ch_address)))(7 downto 0); --write adc data in message buffer
                            cmd_read_mem_answ(2) <= "0000" & adc_data_store(to_integer(unsigned(receive_ch_address)))(11 downto 8);-- 
                        else
                            cmd_read_mem_answ_total(1) <= adc_data_store(0)(7 downto 0); --write adc data in message buffer
                            cmd_read_mem_answ_total(2) <= "0000" & adc_data_store(0)(11 downto 8);-- 
                            
                            cmd_read_mem_answ_total(3) <= adc_data_store(1)(7 downto 0); --write adc data in message buffer
                            cmd_read_mem_answ_total(4) <= "0000" & adc_data_store(1)(11 downto 8);--
                            
                            cmd_read_mem_answ_total(5) <= adc_data_store(2)(7 downto 0); --write adc data in message buffer
                            cmd_read_mem_answ_total(6) <= "0000" & adc_data_store(2)(11 downto 8);--
                            
                            cmd_read_mem_answ_total(7) <= adc_data_store(3)(7 downto 0); --write adc data in message buffer
                            cmd_read_mem_answ_total(8) <= "0000" & adc_data_store(3)(11 downto 8);--
                            
                            cmd_read_mem_answ_total(9) <= adc_data_store(4)(7 downto 0); --write adc data in message buffer
                            cmd_read_mem_answ_total(10) <= "0000" & adc_data_store(4)(11 downto 8);--
                            
                            cmd_read_mem_answ_total(11) <= adc_data_store(5)(7 downto 0); --write adc data in message buffer
                            cmd_read_mem_answ_total(12) <= "0000" & adc_data_store(5)(11 downto 8);--
                            
                            cmd_read_mem_answ_total(13) <= adc_data_store(6)(7 downto 0); --write adc data in message buffer
                            cmd_read_mem_answ_total(14) <= "0000" & adc_data_store(6)(11 downto 8);--
                            
                            cmd_read_mem_answ_total(15) <= adc_data_store(7)(7 downto 0); --write adc data in message buffer
                            cmd_read_mem_answ_total(16) <= "0000" & adc_data_store(7)(11 downto 8);--
                            
                            cmd_read_mem_answ_total(17) <= adc_data_store(8)(7 downto 0); --write adc data in message buffer
                            cmd_read_mem_answ_total(18) <= "0000" & adc_data_store(8)(11 downto 8);--
                            
                        end if;
                        state <= CMD_READ_MEM_ANSWER;
                        read_receive_cnt <= (others => '0');
                    end if;
                    
                    if mscb_out_parallel(10) = '1' then
                        if read_receive_cnt = 7 then --channel address
                            receive_ch_address <= mscb_in_parallel(7 downto 0);
                        elsif read_receive_cnt = 8 then --crc
                            receive_crc <= mscb_in_parallel(7 downto 0);
                        elsif read_receive_cnt = 2 then
                            if mscb_in_parallel(7 downto 0) = x"02" or mscb_in_parallel(7 downto 0) = x"12" then
                                numb_data_bytes <= mscb_in_parallel(7 downto 0);
                            else
                                state <= idle;
                            end if;
                            
                        elsif mscb_in_parallel(7 downto 0) /= cmd_read_mem_adc(6-to_integer(read_receive_cnt)) then 
                            state <= idle;
                        else 
                            full_operation_save(to_integer(read_receive_cnt)) <= mscb_in_parallel; -- is not needed anymore only for debugging
                            --LED1 <= full_operation_save(to_integer(read_receive_cnt))(0) and full_operation_save(to_integer(read_receive_cnt))(1) and full_operation_save(to_integer(read_receive_cnt))(2) and full_operation_save(to_integer(read_receive_cnt))(3) and full_operation_save(to_integer(read_receive_cnt))(4) and full_operation_save(to_integer(read_receive_cnt))(5) and full_operation_save(to_integer(read_receive_cnt))(6) and full_operation_save(to_integer(read_receive_cnt))(7);                       
                        end if;
                        read_receive_cnt <= read_receive_cnt + 1;
                        mscb_out_parallel(10) <= '0';                        
                    end if;
                 
                when CMD_READ_MEM_ANSWER =>
                   
                        
                    if (read_mem_answer_cnt = 6 and numb_data_bytes=x"02") or (read_mem_answer_cnt = 22 and numb_data_bytes=x"12") then 
                        read_mem_answer_cnt <= (others => '0');
                        state <= idle;
                        mscb_out_parallel(10 downto 0) <= "00000000000";

                    elsif numb_data_bytes=x"02" then
                        mscb_out_parallel(10 downto 0) <= "010" & cmd_read_mem_answ(5-to_integer(read_mem_answer_cnt));
                    elsif numb_data_bytes=x"12" then
                        mscb_out_parallel(10 downto 0) <= "010" & cmd_read_mem_answ_total(21-to_integer(read_mem_answer_cnt));
                    end if;
                    
                    if mscb_out_parallel(9) = '1' then
                        mscb_out_parallel(10 downto 0) <= (others => '0');
                        read_mem_answer_cnt <= read_mem_answer_cnt +1;
                    end if;
                        
                    
                when others =>
                    state <= idle;
            end case;
        end if;
    end process;

--    sw(0)   <= SWITCH1;
----	sw(1)   <= SWITCH2;
----	sw(2)   <= SWITCH3;
----    sw(3)   <= SWITCH4;
----    sw(4)   <= SWITCH5;
--    
--      LED1    <= full_operation_save(0)(0);
--    LED2    <= e_x(0);
--    LED3    <= e_x2(0);
--    LED4    <= counter(0);
--    LED5    <= led(0);

    
--    process(adc_clk,reset_n)
--    begin
--        if (reset_n='0') then
--            e_x <= (others => '0');
--            e_x2 <= (others => '0');
--            counter <= (others => '0');
--            
--        elsif rising_edge(adc_clk) then
--        
--            if mult_delay_cnt = 2 then 
--                e_x2 <= e_x2 + unsigned(x2);
--            end if;  
--            
--            if (counter < count_max) then
--                mult_delay_cnt <= mult_delay_cnt + 1;
--                led(0) <= '1';
--                if adc_response_valid = '1' then
--                    mult_delay_cnt <= (others => '0');
--                    e_x <= e_x + unsigned(adc_data);
--                    counter <= counter + 1;
--                end if;  
--            else
--            
--                led(0) <= '0';
--                
--            end if;
--        end if;
--    end process;
--
--    --- Multiplier ---
--    e_mult : my_altmult_add
--	port map (
--		
--			clock0		=> adc_clk,
--			dataa_0		=> adc_data,
--			datab_0		=> adc_data,
--			result		=> x2
--	);
--
-- 
    ---ADC---
    testadc : component work.cmp.myadc
    port map(
			adc_pll_clock_clk      => adc_clk,
			adc_pll_locked_export  => pll_locked,           -- export
			clock_clk              => adc_clk,             -- clk
			command_valid          => '1',                -- valid
			command_channel        => adc_channels(to_integer(adc_cnt)),              -- channel
			command_startofpacket  => '1',                -- startofpacket
			command_endofpacket    => '1',                -- endofpacket
			command_ready          => open,                 -- ready
			reset_sink_reset_n     => reset_n,              -- reset_n
			response_valid         => adc_response_valid,   -- valid
			response_channel       => response_channel,                 -- channel
			response_data          => adc_data,             -- data
			response_startofpacket => open,                 -- startofpacket
			response_endofpacket   => open--,                 -- endofpacket
		);


	--- PLL ---
	e_ip_altpll : entity work.ip_altpll
	port map (
		areset => not reset_n,
		inclk0 => CLOCK,
		c0     => adc_clk,
		c1     => nios_clk,
        c2     => slow_clk,
		locked => pll_locked--,
	);
    
       
        --    process(adc_clk, reset_n)
--    begin
--        if (reset_n ='0') then
--            led <= (others => '0');
--            adc_data_store <= (others => '0');
--        elsif rising_edge(adc_clk) then 
--            if adc_response_valid = '1' then
--                adc_data_store <= adc_data;
--            end if; 
--            if (sw(1) = '1') then
--                led <=  adc_data_store(4 downto 0);
--            elsif ( sw(2) = '1') then
--                led <= adc_data_store(9 downto 5);
--            else 
--                led <= "000" & adc_data_store(11 downto 10);
--            end if;
--        end if;
--    end process;
----    
--    channel <= "10001" when SWITCH4='1' else "00001";
--
-- 	   

--    	--- NIOS ---
--	e_nios : component work.cmp.nios
--	port map (
--			clk_clk           => CLOCK,
--			led_io_export     => led,
--			pio_export        => nios_pio,
--			rst_reset_n       => reset_n,
--			spi_MISO          => Arduino_IO13,
--			spi_MOSI          => Arduino_IO12,
--			spi_SCLK          => Arduino_IO11,
--			spi_SS_n          => Arduino_IO10,
--			sw_io_export      => sw--,
--	);

end architecture;
