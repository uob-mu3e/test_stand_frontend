library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;

use work.mudaq.all;

entity time_merger_tree_fifo_64 is
generic (
    TREE_w          : positive  := 8;
    TREE_r          : positive  := 8;
    r_width         : integer   := 64;
    w_width         : integer   := 64;
    compare_fifos   : integer   := 32;
    gen_fifos       : integer   := 16--;
);
port (
    -- input 
    i_fifo_q        : in    work.util.slv76_array_t(compare_fifos - 1 downto 0);
    i_fifo_empty    : in std_logic_vector(compare_fifos - 1 downto 0);
    i_fifo_ren      : in std_logic_vector(gen_fifos - 1 downto 0);
    i_merge_state   : in std_logic;
    i_mask_n        : in std_logic_vector(compare_fifos - 1 downto 0);

    -- output
    o_fifo_q        : out   work.util.slv76_array_t(gen_fifos - 1 downto 0);
    o_fifo_empty    : out std_logic_vector(gen_fifos - 1 downto 0);
    o_fifo_ren      : out std_logic_vector(compare_fifos - 1 downto 0);
    o_mask_n        : out std_logic_vector(gen_fifos - 1 downto 0);
    
    i_reset_n       : in    std_logic;
    i_clk           : in    std_logic--;
);
end entity;

architecture arch of time_merger_tree_fifo_64 is

    -- merger signals
    constant size : integer := compare_fifos/2;
    
    signal fifo_data, fifo_data_reg, fifo_q, fifo_q_reg : work.util.slv76_array_t(gen_fifos - 1 downto 0);
    signal layer_state : work.util.slv4_array_t(gen_fifos - 1 downto 0);
    signal fifo_ren_reg, fifo_ren : std_logic_vector(compare_fifos - 1 downto 0);
    signal fifo_wen, fifo_wen_reg, fifo_full, fifo_full_reg, fifo_empty, fifo_empty_reg, reset_fifo, reset_fifo_reg0, reset_fifo_reg1, reset_fifo_reg2 : std_logic_vector(gen_fifos - 1 downto 0);


begin

    o_fifo_ren      <= fifo_ren;
    o_fifo_q        <= fifo_q;
    o_fifo_empty    <= fifo_empty;

    tree_fifos:
    FOR i in 0 to gen_fifos - 1 GENERATE

    o_mask_n(i) <= i_mask_n(i) or i_mask_n(i + size);

    e_link_fifo : entity work.ip_dcfifo_mixed_widths
    generic map(
        ADDR_WIDTH_w    => TREE_w,
        DATA_WIDTH_w    => w_width,
        ADDR_WIDTH_r    => TREE_r,
        DATA_WIDTH_r    => r_width,
        DEVICE          => "Arria 10"--,
    )
    port map (
        aclr    => not i_reset_n or reset_fifo(i),
        data    => fifo_data_reg(i),
        rdclk   => i_clk,
        rdreq   => i_fifo_ren(i),
        wrclk   => i_clk,
        wrreq   => fifo_wen_reg(i),
        q       => fifo_q_reg(i),
        rdempty => fifo_empty_reg(i),
        rdusedw => open,
        wrfull  => fifo_full_reg(i),
        wrusedw => open--,
    );

    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        fifo_ren(i) <= '0';
        fifo_ren(i + size) <= '0';
        fifo_ren_reg(i) <= '0';
        fifo_ren_reg(i + size) <= '0';
        fifo_wen(i) <= '0';
        fifo_data(i) <= (others => '0');
        layer_state(i) <= (others => '0');
        reset_fifo(i) <= '0';
        reset_fifo_reg0(i) <= '0';
        reset_fifo_reg1(i) <= '0';
        reset_fifo_reg2(i) <= '0';
        --
    elsif rising_edge(i_clk) then
        fifo_ren(i) <= '0';
        fifo_ren(i + size) <= '0';
        fifo_ren_reg(i) <= fifo_ren(i);
        fifo_ren_reg(i + size) <= fifo_ren(i + size);
        reset_fifo(i) <= '0';
        reset_fifo_reg0(i) <= reset_fifo(i);
        reset_fifo_reg1(i) <= reset_fifo_reg0(i);
        reset_fifo_reg2(i) <= reset_fifo_reg1(i);
        fifo_wen(i) <= '0';
        
        -- 31 downto 0  -> hit1
        -- 31 downto 28 -> ts1
        -- 37 downto 32 -> link number1
        -- 69 downto 38 -> hit2
        -- 69 downto 66 -> ts2
        -- 75 downto 70 -> link number 2
        if ( i_merge_state = '1' ) then
            case layer_state(i) is
            
                when "0000" =>
                    if ( fifo_full(i) = '1' or reset_fifo(i) = '1' ) then -- or reset_fifo_reg0(i) = '1' or reset_fifo_reg1(i) = '1' or reset_fifo_reg2(i) = '1' ) then
                        --
                    elsif ( i_mask_n(i) = '0' or i_mask_n(i + size) = '0' ) then
                        layer_state(i) <= "1111";
                    elsif ( i_fifo_empty(i) = '1' or i_fifo_empty(i + size) = '1' or fifo_ren(i) = '1' or fifo_ren(i + size) = '1' or fifo_ren_reg(i) = '1' or fifo_ren_reg(i + size) = '1' ) then
                        --
                    else
                        -- TODO: define signal for empty since the fifo should be able to get empty if no hits are comming
                        if ( i_fifo_q(i)(31 downto 28) <= i_fifo_q(i + size)(31 downto 28) and i_fifo_q(i)(37 downto 0) /= tree_padding ) then
                            fifo_data(i)(37 downto 0) <= i_fifo_q(i)(37 downto 0);
                            layer_state(i)(0) <= '1';
                            if ( i_fifo_q(i)(69 downto 66) <= i_fifo_q(i + size)(31 downto 28) and i_fifo_q(i)(75 downto 38) /= tree_zero and i_fifo_q(i)(75 downto 38) /= tree_padding ) then
                                fifo_data(i)(75 downto 38) <= i_fifo_q(i)(75 downto 38);
                                layer_state(i)(1) <= '1';
                                fifo_wen(i) <= '1';
                                fifo_ren(i) <= '1';
                            elsif ( i_fifo_q(i + size)(37 downto 0) /= tree_padding ) then
                                fifo_data(i)(75 downto 38) <= i_fifo_q(i + size)(37 downto 0);
                                layer_state(i)(2) <= '1';
                                fifo_wen(i) <= '1';
                            elsif ( i_fifo_q(i)(75 downto 38) = tree_padding and i_fifo_q(i + size)(37 downto 0) = tree_padding ) then
                                fifo_data(i)(75 downto 38) <= tree_padding;
                                fifo_wen(i) <= '1';
                                layer_state(i) <= "1110";
                            end if;
                        elsif ( i_fifo_q(i + size)(37 downto 0) /= tree_padding ) then
                            fifo_data(i)(37 downto 0) <= i_fifo_q(i + size)(37 downto 0);
                            layer_state(i)(2) <= '1';
                            if ( i_fifo_q(i)(31 downto 28) <= i_fifo_q(i + size)(69 downto 66) and i_fifo_q(i)(37 downto 0) /= tree_padding ) then
                                fifo_data(i)(75 downto 38) <= i_fifo_q(i)(37 downto 0);
                                layer_state(i)(0) <= '1';
                                fifo_wen(i) <= '1';
                            elsif ( i_fifo_q(i + size)(75 downto 38) /= tree_zero and i_fifo_q(i + size)(75 downto 38) /= tree_padding ) then
                                fifo_data(i)(75 downto 38) <= i_fifo_q(i + size)(75 downto 38);
                                layer_state(i)(3) <= '1';
                                fifo_wen(i) <= '1';
                                fifo_ren(i + size) <= '1';
                            elsif ( i_fifo_q(i)(37 downto 0) = tree_padding and i_fifo_q(i + size)(75 downto 38) = tree_padding ) then
                                fifo_data(i)(75 downto 38) <= tree_padding;
                                fifo_wen(i) <= '1';
                                layer_state(i) <= "1110";
                            end if;
                        elsif ( i_fifo_q(i)(37 downto 0) = tree_padding and i_fifo_q(i + size)(37 downto 0) = tree_padding ) then 
                            fifo_data(i) <= tree_padding & tree_padding;
                            fifo_wen(i) <= '1';
                            layer_state(i) <= "1110";
                        end if;
                    end if;
                when "0011" =>
                    layer_state(i) <= (others => '0');
                    -- TODO: reg of ren probably not needed
                    fifo_ren_reg(i) <= '1';
                    fifo_ren_reg(i + size) <= '1';
                when "1100" =>
                    layer_state(i) <= (others => '0');
                    -- TODO: reg of ren probably not needed
                    fifo_ren_reg(i) <= '1';
                    fifo_ren_reg(i + size) <= '1';
                when "0101" =>
                    if ( i_fifo_q(i)(69 downto 66) <= i_fifo_q(i + size)(69 downto 66) and i_fifo_q(i)(75 downto 38) /= tree_zero and i_fifo_q(i)(75 downto 38) /= tree_padding ) then
                        fifo_data(i)(37 downto 0) <= i_fifo_q(i)(75 downto 38);
                        layer_state(i)(0) <= '0';
                        fifo_ren(i) <= '1';
                    elsif ( i_fifo_q(i + size)(75 downto 38) /= tree_zero and i_fifo_q(i + size)(75 downto 38) /= tree_padding ) then
                        fifo_data(i)(37 downto 0) <= i_fifo_q(i + size)(75 downto 38);
                        layer_state(i)(2) <= '0';
                        fifo_ren(i + size) <= '1';
                    elsif ( i_fifo_q(i)(75 downto 38) = tree_padding and i_fifo_q(i + size)(75 downto 38) = tree_padding ) then
                        fifo_data(i) <= tree_padding & tree_padding;
                        fifo_wen(i) <= '1';
                        layer_state(i) <= "1110";
                    end if;
                when "0100" =>
                    -- TODO: define signal for empty since the fifo should be able to get empty if no hits are comming
                    if ( i_fifo_empty(i) = '0' and fifo_ren(i) = '0' and fifo_ren_reg(i) = '0' ) then
                        -- TODO: what to do when i_fifo_q(i + size)(69 downto 66) is zero? maybe error cnt?
                        if ( i_fifo_q(i)(31 downto 28) <= i_fifo_q(i + size)(69 downto 66) and i_fifo_q(i)(37 downto 0) /= tree_padding ) then
                            fifo_data(i)(75 downto 38) <= i_fifo_q(i)(37 downto 0);
                            layer_state(i)(0) <= '1';
                            fifo_wen(i) <= '1';
                        elsif ( i_fifo_q(i + size)(75 downto 38) /= tree_zero and i_fifo_q(i + size)(75 downto 38) /= tree_padding ) then
                            fifo_data(i)(75 downto 38) <= i_fifo_q(i + size)(75 downto 38);
                            layer_state(i)(3) <= '1';
                            fifo_wen(i) <= '1';
                            fifo_ren(i + size) <= '1';
                        elsif ( i_fifo_q(i)(37 downto 0) = tree_padding and i_fifo_q(i + size)(75 downto 38) = tree_padding ) then
                            fifo_data(i)(75 downto 38) <= tree_padding;
                            fifo_wen(i) <= '1';
                            layer_state(i) <= "1110";
                        end if;
                    else
                        -- TODO: wait for fifo_0 i here --> error counter?
                    end if;
                when "0001" =>
                    -- TODO: define signal for empty since the fifo should be able to get empty if no hits are comming
                    if ( i_fifo_empty(i + size) = '0' and fifo_ren(i + size) = '0' and fifo_ren_reg(i + size) = '0' ) then       
                        -- TODO: what to do when i_fifo_q(i)(69 downto 66) is zero? maybe error cnt?     
                        if ( i_fifo_q(i)(69 downto 66) <= i_fifo_q(i + size)(31 downto 28) and i_fifo_q(i)(75 downto 38) /= tree_zero and i_fifo_q(i)(75 downto 38) /= tree_padding ) then
                            fifo_data(i)(75 downto 38) <= i_fifo_q(i)(75 downto 38);
                            layer_state(i)(1) <= '1';
                            fifo_wen(i) <= '1';
                            fifo_ren(i) <= '1';
                        elsif ( i_fifo_q(i + size)(37 downto 0) /= tree_padding ) then
                            fifo_data(i)(75 downto 38) <= i_fifo_q(i + size)(37 downto 0);
                            layer_state(i)(2) <= '1';
                            fifo_wen(i) <= '1';
                        elsif ( i_fifo_q(i + size)(37 downto 0) = tree_padding and i_fifo_q(i)(75 downto 38) = tree_padding ) then
                            fifo_data(i)(75 downto 38) <= tree_padding;
                            fifo_wen(i) <= '1';
                            layer_state(i) <= "1110";
                        end if;
                    else
                        -- TODO: wait for fifo_0 i+size here --> error counter?
                    end if;
                when "1111" =>
                    if ( fifo_full(i) = '0' ) then
                        if ( i_mask_n(i) = '1' and i_fifo_empty(i) = '0' and fifo_ren(i) = '0' and fifo_ren_reg(i) = '0' and i_fifo_q(i)(75 downto 38) /= tree_zero ) then 
                            fifo_data(i) <= i_fifo_q(i);    
                            fifo_wen(i) <= '1';
                            fifo_ren(i) <= '1';
                        elsif ( i_mask_n(i + size) = '1' and i_fifo_empty(i + size) = '0' and fifo_ren(i + size) = '0' and fifo_ren_reg(i + size) = '0' and i_fifo_q(i + size)(75 downto 38) /= tree_zero ) then
                            fifo_data(i) <= i_fifo_q(i + size);    
                            fifo_wen(i) <= '1';
                            fifo_ren(i + size) <= '1';
                        elsif ( i_mask_n(i) = '0' and i_mask_n(i + size) = '0' ) then
                            fifo_data(i) <= tree_padding & tree_padding;
                            layer_state(i) <= "1110";
                            fifo_wen(i) <= '1';
                        end if;
                    end if;
                when "1110" =>
                    -- TODO: write out something?
                when others =>
                    layer_state(i) <= (others => '0');

            end case;
        else
            reset_fifo(i) <= '1';
            layer_state(i) <= "0000";
            fifo_data(i) <= (others => '0');
        end if;
    end if;
    end process;
    
    -- reg for FIFO outputs (timing)
    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        fifo_q(i) <= (others => '0');
        fifo_empty(i) <= '0';
        --
    elsif rising_edge(i_clk) then
        fifo_q(i) <= fifo_q_reg(i);
        fifo_empty(i) <= fifo_empty_reg(i);
    end if;
    end process;
    
    -- reg for FIFO inputs (timing)
    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        fifo_data_reg(i)    <= (others => '0');
        fifo_wen_reg(i) <= '0';
        fifo_full(i)    <= '1';
    elsif rising_edge(i_clk) then
        fifo_data_reg(i)    <= fifo_data(i);
        fifo_wen_reg(i)     <= fifo_wen(i);
        fifo_full(i)        <= fifo_full_reg(i);
    end if;
    end process;
    
    END GENERATE tree_fifos;

end architecture;
