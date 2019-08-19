library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.daq_constants.all;

ENTITY mscb is
generic (
    CLK_FREQ_g : positive := 125000000--;
);
port (
    nios_clk                    : in    std_logic;
    reset                       : in    std_logic;
    mscb_to_nios_parallel_in    : out   std_logic_vector(11 downto 0);
    mscb_from_nios_parallel_out : in    std_logic_vector(11 downto 0);
    mscb_data_in                : in    std_logic;
    mscb_data_out               : out   std_logic;
    mscb_oe                     : out   std_logic;
    mscb_counter_in             : out   std_logic_vector(15 downto 0);
    o_mscb_irq                  : out   std_logic--;
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

    mscb_nios_out                           <= mscb_from_nios_parallel_out(8 downto 0);
    DataReady                               <= not out_fifo_empty;

------------- Wire up components --------------------

    e_slow_counter : entity work.counter
    generic map (
        W           => 16,
        DIV         => 100--,
    )
    port map(
        cnt         => mscb_counter_in,
        ena         => '1',
        reset       => reset,
        clk         => clk--,
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


    e_fifo_data_in : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH => 8,
        DATA_WIDTH => 9--,
    )
    port map (
        sclr            => reset,
        clock           => clk,
        data            => uart_parallel_out,
        rdreq           => in_fifo_read_request,
        wrreq           => in_fifo_write_request,
        empty           => in_fifo_empty,
        full            => in_fifo_full,
        q               => in_fifo_data_out,
        usedw           => open,
        almost_full     => open,
        almost_empty    => open--,
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

    e_fifo_data_out : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH => 8,
        DATA_WIDTH => 9--,
    )
    port map (
        sclr            => reset,
        clock           => clk,
        data            => mscb_nios_out, 
        rdreq           => out_fifo_read_request,
        wrreq           => out_fifo_write_request,
        empty           => out_fifo_empty,
        full            => out_fifo_full,
        q               => out_fifo_data_out,
        usedw           => open,
        almost_full     => open,
        almost_empty    => open--,
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
