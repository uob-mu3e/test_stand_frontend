library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.daq_constants.all;

ENTITY mscb is
generic (
    CLK_FREQ_g : positive := 50000000--;
);
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
END ENTITY;

architecture rtl of mscb is

------------------ Signal declaration ------------------------

    -- external connections
    signal clk :                        std_logic;

    -- mscb data flow
    signal uart_generated_data :        std_logic;
    signal signal_in :                  std_logic;
    signal signal_out :                 std_logic;
    signal uart_serial_in :             std_logic;
    signal uart_read_enable :           std_logic;
    signal uart_parallel_out :          std_logic_vector(8 downto 0);

    signal in_fifo_read_request :       std_logic;
    signal in_fifo_write_request :      std_logic;
    signal in_fifo_empty :              std_logic;
    signal in_fifo_full :               std_logic;
    signal in_fifo_data_out :           std_logic_vector(8 downto 0);

    signal out_fifo_read_request :      std_logic;
    signal out_fifo_write_request :     std_logic;
    signal out_fifo_empty :             std_logic;
    signal out_fifo_full :              std_logic;
    signal out_fifo_data_out :          std_logic_vector(8 downto 0);

    signal DataReady :                  std_logic;

    signal mscb_nios_out :              std_logic_vector(8 downto 0);
    signal mscb_data_ready :            std_logic;

    signal DataGeneratorEnable :        std_logic;
    signal uart_serial_out :            std_logic; --uart data for the output pin

    signal Transmitting  :              std_logic; -- uart data is being send
    signal dummy :                      std_logic;
    
    signal addressing_data_in :         std_logic_vector(8 downto 0);
    signal addressing_data_out :        std_logic_vector(8 downto 0);
    signal addressing_wrreq :           std_logic;
    signal addressing_rdreq :           std_logic;
    signal rec_fifo_empty :             std_logic;

----------------------------------------
------------ Begin top level------------
----------------------------------------

begin

--------- external input connections -----------

  clk <= nios_clk;

------------- external input/output connections -----------------

  --i/o switches
    process(Transmitting, reset, Clk)
    begin
    if (Transmitting = '1') then
        mscb_data_out   <= uart_serial_out;
        mscb_oe         <= '1';
    else
        mscb_data_out   <= 'Z';
        --mscb_oe       <= '0';       -- single FPGA connected to converter chip
        mscb_oe         <= 'Z';         -- multiple FPGAs 
    end if;
    end process;

    --hsma_d(0) <= 'Z' when Transmitting = '1' else 'Z';
    signal_in <= '1' when Transmitting = '1' else (mscb_data_in);

---------------internal connections-----------------------



    ---- parallel in for the nios ----
    mscb_to_nios_parallel_in(8 downto 0)    <= in_fifo_data_out; -- 8+1 bit mscb words
    mscb_to_nios_parallel_in(9)             <= in_fifo_empty;
    mscb_to_nios_parallel_in(10)            <= in_fifo_full;
    mscb_to_nios_parallel_in(11)            <= '1';
    
    ---- interrupt to nios ----
    o_mscb_irq                              <= not in_fifo_empty;

    mscb_nios_out                           <= mscb_from_nios_parallel_out(8 downto 0);
    DataReady                               <= not out_fifo_empty;

------------- Wire up components --------------------

    e_slow_counter : entity work.counter_async
    port map(
        CounterOut(15 downto 0)  => mscb_counter_in,
        Enable                   => '1',
        Reset                    => reset,
        Clk                      => clk,
        CountDown                => '0',
        Init                     => to_unsigned(0,32)
    );

  -- wire up uart reciever for mscb
    e_uart_rx : entity work.uart_reciever
    generic map (
        Clk_Ratio   => CLK_FREQ_g / 115200--,
    )
    port map (
        Clk         => clk,
        Reset       => reset,
        DataIn      => uart_serial_in,
        ReadEnable  => uart_read_enable,
        DataOut     => uart_parallel_out
    );

    -- fifo addressing to nios (addressing is done, only commands intended for this mscb node)
    e_nios_fifo : entity work.my_fifo
    port map (
        sclr            => reset,
        clock           => clk,
        data            => addressing_data_out,
        rdreq           => in_fifo_read_request,
        wrreq           => addressing_wrreq,
        empty           => in_fifo_empty,
        full            => in_fifo_full,
        q               => in_fifo_data_out
    );

    -- fifo receiver to addressing
    e_fifo_data_in : entity work.my_fifo
    port map (
        sclr            => reset,
        clock           => clk,
        data            => uart_parallel_out,
        rdreq           => addressing_rdreq,
        wrreq           => in_fifo_write_request,
        empty           => rec_fifo_empty,
        full            => open,
        q               => addressing_data_in
    );

    -- wire up uart transmitter for mscb
    e_uart_tx : entity work.uart_transmitter
    generic map (
        Clk_Ratio   => CLK_FREQ_g / 115200--,
    )
    port map (
        Clk             => clk,
        Reset           => reset,
        DataIn          => out_fifo_data_out,
        DataReady       => DataReady,
        ReadRequest     => out_fifo_read_request,
        DataOut         => uart_serial_out,
        Transmitting    => Transmitting
    );

    -- fifo nios to transmitter
    e_fifo_data_out : entity work.my_fifo
    port map (
        sclr            => reset,
        clock           => clk,
        data            => mscb_nios_out, 
        rdreq           => out_fifo_read_request,
        wrreq           => out_fifo_write_request,
        empty           => out_fifo_empty,
        full            => out_fifo_full,
        q               => out_fifo_data_out
    );


    -- make the fifo read request from the nios one clocktick long
    e_uEdgeFIFORead : entity work.edge_detector
    port map (
        clk         => clk,
        signal_in   => mscb_from_nios_parallel_out(10),
        output      => in_fifo_read_request
    );

    -- make the fifo write request from the nios one clocktick long
    e_uEdgeFIFOWrite : entity work.edge_detector
    port map (
        clk             => clk,
        signal_in       => mscb_from_nios_parallel_out(9),
        output          => out_fifo_write_request
    );
    

    e_mscb_addressing : entity work.mscb_addressing
    port map (
        i_clk           => clk,
        i_reset         => reset,
        i_data          => addressing_data_in,
        i_empty         => rec_fifo_empty,
        i_address       => i_mscb_address,
        o_data          => addressing_data_out,
        o_wrreq         => addressing_wrreq,
        o_rdreq         => addressing_rdreq
    );


    process(Clk, reset)
    begin
        if reset = '1' then
            DataGeneratorEnable         <= '0';
            --uart_serial_in            <='1';
            --hsma_d(0)                 <='Z';
        elsif rising_edge(clk) then
            DataGeneratorEnable         <= '1';
            in_fifo_write_request       <= uart_read_enable;
            uart_serial_in              <= signal_in;
        end if;
    end process;

end architecture;
