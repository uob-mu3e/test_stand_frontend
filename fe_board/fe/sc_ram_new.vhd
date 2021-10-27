--
-- SC rewrite
-- Oktober 2021, M.Mueller
--

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_misc.all;
use work.mudaq.all;

entity sc_ram_new is
generic (
    RAM_ADDR_WIDTH_g : positive := 16;
    READ_DELAY_g     : positive := 6--;
);
port (
    -- RAM port (slave of sx_rx)
    i_ram_addr          : in    std_logic_vector(15 downto 0) := (others => '0');
    i_ram_re            : in    std_logic := '0';
    o_ram_rvalid        : out   std_logic;
    o_ram_rdata         : out   std_logic_vector(31 downto 0);
    i_ram_we            : in    std_logic := '0';
    i_ram_wdata         : in    std_logic_vector(31 downto 0) := (others => '0');

    -- AVS port (avalon slave of nios)
    -- address units - words
    i_avs_address       : in    std_logic_vector(15 downto 0) := (others => '0');
    i_avs_read          : in    std_logic := '0';
    o_avs_readdata      : out   std_logic_vector(31 downto 0);
    i_avs_write         : in    std_logic := '0';
    i_avs_writedata     : in    std_logic_vector(31 downto 0) := (others => '0');
    o_avs_waitrequest   : out   std_logic;

    -- REG port (master)
    o_reg_addr          : out   std_logic_vector(15 downto 0);
    o_reg_re            : out   std_logic;
    i_reg_rdata         : in    std_logic_vector(31 downto 0) := (others => '0');
    o_reg_we            : out   std_logic;
    o_reg_wdata         : out   std_logic_vector(31 downto 0);

    i_reset_n           : in    std_logic;
    i_clk               : in    std_logic--;
);
end entity;

architecture arch of sc_ram_new is

    signal iram_addr    : std_logic_vector(15 downto 0);
    signal iram_we      : std_logic;
    signal iram_rdata   : std_logic_vector(31 downto 0); 
    signal iram_wdata   : std_logic_vector(31 downto 0);

    signal read_delay_shift_reg         : std_logic_vector(READ_DELAY_g downto 0) := (others => '0');
    signal read_delay_shift_reg_type    : std_logic_vector(READ_DELAY_g downto 0) := (others => '0'); -- 0: Arria, 1: Nios

    signal internal_ram_return_queue    : reg32array(READ_DELAY_g downto 0);

    signal avs_waitrequest : std_logic;
    signal avs_cmd_buf     : std_logic := '0';

    -- signals to lvl0 sc_node
    signal addr          : std_logic_vector(15 downto 0) := (others => '0');
    signal re            : std_logic := '0';
    signal rdata         : std_logic_vector(31 downto 0) := (others => '0');
    signal we            : std_logic := '0';
    signal wdata         : std_logic_vector(31 downto 0) := (others => '0');

begin
 
    -- request process
    process(i_clk, i_reset_n)
    begin
        if ( i_reset_n = '0' ) then
            re              <= '0';
            we              <= '0';
            read_delay_shift_reg        <= (others => '0');
            read_delay_shift_reg_type   <= (others => '0');

        elsif rising_edge(i_clk) then
            -- defaults
            we     <= '0';
            re     <= '0';

            read_delay_shift_reg        <= read_delay_shift_reg(READ_DELAY_g-1 downto 0) & '0';
            read_delay_shift_reg_type   <= read_delay_shift_reg_type(READ_DELAY_g-1 downto 0) & '0';

            if(i_ram_we='1') then -- write from Arria10
                we      <= '1';
                wdata   <= i_ram_wdata;
                addr    <= i_ram_addr;
            elsif(i_ram_re='1') then -- read from Arria10
                read_delay_shift_reg(0) <= '1';
                re      <= '1';
                addr    <= i_ram_addr;
            elsif(i_avs_write='1') then -- write from nios
                we      <= '1';
                wdata   <= i_avs_writedata;
                addr    <= i_avs_address;
            elsif(i_avs_read='1') then -- read from nios
                read_delay_shift_reg(0)         <= '1';
                read_delay_shift_reg_type(0)    <= '1';
                re      <= '1';
                addr    <= i_avs_address;
            end if;
        end if;
    end process;


    -- response process
    process(i_clk, i_reset_n)
    begin
        if ( i_reset_n = '0' ) then
            avs_cmd_buf     <= '0';
            o_avs_readdata  <= (others => '0');

        elsif rising_edge(i_clk) then
            -- defaults
            o_ram_rvalid    <= '0';
            avs_cmd_buf     <= i_avs_read or i_avs_write;

            -- delay internal ram by the same amount of cycles as the sc regs
            -- (avoids collisions between read response from internal ram and sc reg)
            internal_ram_return_queue <= internal_ram_return_queue(READ_DELAY_g-1 downto 0) & iram_rdata;

            if(read_delay_shift_reg(READ_DELAY_g) = '1') then
                if(read_delay_shift_reg_type(READ_DELAY_g) = '0') then --respond to Arria10
                    o_ram_rvalid    <= '1';
                    o_ram_rdata     <= i_reg_rdata;
                else -- respond to nios
                    o_avs_readdata  <= i_reg_rdata;
                    -- TODO: search for correct avm_waitrequest timing
                end if;
            end if;
        end if;
    end process;

    o_avs_waitrequest <= avs_waitrequest;
    avs_waitrequest <=
        '1' when ( i_ram_re = '1' 
                or i_ram_we = '1' 
                or (or_reduce(read_delay_shift_reg_type) = '1' and read_delay_shift_reg_type(READ_DELAY_g) = '0')
                or (avs_cmd_buf = '0' and (i_avs_read = '1' or i_avs_write = '1')))
            else '0';


    lvl0_sc_node: entity work.sc_node
    generic map (
        ADD_SLAVE1_DELAY_g  => 6,
        N_REPLY_CYCLES_g    => 6,
        SLAVE0_ADDR_MATCH_g => "11111111--------"
    )
    port map (
        i_clk          => i_clk,
        i_reset_n      => i_reset_n,
        
        i_master_addr  => addr,
        i_master_re    => re,
        o_master_rdata => rdata,
        i_master_we    => we,
        i_master_wdata => wdata,

        o_slave0_addr  => o_reg_addr,
        o_slave0_re    => o_reg_re,
        i_slave0_rdata => i_reg_rdata,
        o_slave0_we    => o_reg_we,
        o_slave0_wdata => o_reg_wdata,

        o_slave1_addr  => iram_addr,
        o_slave1_re    => open,
        i_slave1_rdata => iram_rdata,
        o_slave1_we    => iram_we,
        o_slave1_wdata => iram_wdata
    );

    --1: nios avm needs to wait when: 
            -- sc_rx wants to read something right now (*)
            -- sc_rx wants to write something right now
            -- there is a read of the nios in the queue (or_reduce(...)), but it has not arrived yet (it is not at position READ_DELAY_g)
            -- nios wants to put something into the queue now (rising edge on i_avs_read or i_avs_write)
            
            -- (*): -- TODO: this is not save, we might lose a valid word in internal_ram_return_queue(READ_DELAY_g) here ... do something about it
            --      -- can we do something about it ? 
            --      -- avm waitrequest does two things:
                        --1: permission to send the next read/write and 
                        --2: sign that the read data is here and should be read now
            --      -- but here we need something that does only 2.
            --      -- buffer avm read return in this case ? .. it should just be max. one word
            --
            -- somthing like 
            -- when read_delay_shift_reg_type(READ_DELAY_g) = '1' and i_ram_re or we = '1' (reply for nios arrives but we cannot allow next rw from nios right now)
            -- then buffer_word <= internal_ram_return_queue / i_reg_rdata
            -- and next time we can reply and allow next rw reply with buffer word instead of internal_ram_return_queue/i_reg_rdata to nios

            -- OR:
            -- export read_data_valid from avm and find out how to do
                --2: sign that the read data is here and should be read now
                -- but not 1: permission to send the next read/write
            -- in av protocol (this is possible ...)


    -- internal RAM
    e_iram : entity work.ram_1r1w
    generic map (
        g_DATA_WIDTH => 32,
        g_ADDR_WIDTH => RAM_ADDR_WIDTH_g--,
    )
    port map (
        i_raddr => iram_addr(RAM_ADDR_WIDTH_g-1 downto 0),
        o_rdata => iram_rdata,
        i_rclk => i_clk,

        i_waddr => iram_addr(RAM_ADDR_WIDTH_g-1 downto 0),
        i_wdata => iram_wdata,
        i_we => iram_we,
        i_wclk => i_clk--,
    );
    
end architecture;
