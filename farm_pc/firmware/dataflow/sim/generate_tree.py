

out_string = ""

for i, layer in enumerate([4, 8, 16, 32]):
    out_string += "when \"{}\" =>".format("".join(["0" for i in range(layer)]))
    out_string += "\n"
    out_string += "-- TODO: define signal for empty since the fifo should be able to get empty if no hits are comming"
    out_string += "\n"
    out_string += "if ( fifo_q_{}(i)(31 downto 28) <= fifo_q_0(i + size{})(31 downto 28) and fifo_empty_{}(i) = '0' and fifo_ren_{}(i) = '0' ) then".format(*[i for j in range(4)])
    out_string += "\n"
    out_string += "fifo_data_{}(i)(31 downto 0) <= fifo_q_{}(i)(31 downto 0);".format(i+1,i)
    out_string += "\n"
    out_string += "layer_{}_state(i)(0) <= '1';".format(i+1)
    out_string += "\n"
    for j in range(int(layer/4)):
        range_0 = (64 + 32*j)-1
        range_1 = (32)-1
        formate_l = [i, range_0, range_0-3, i, i, range_1, range_1-3, i, range_0, range_0-3]
        out_string += "if ( fifo_q_{}(i)({} downto {}) <= fifo_q_{}(i + size{})({} downto {}) and fifo_q_{}(i)({} downto {}) /= x\"00000000\" ) then".format(*formate_l)
        out_string += "\n"
        out_string += "layer_{}_state(i)({}) <= '1';".format(i+1, j+1)
        out_string += "\n"

print(out_string)

"""   
    if ( fifo_q_0(i)(31 downto 28) <= fifo_q_0(i + size1)(31 downto 28) and fifo_empty_0(i) = '0' and fifo_ren_0(i) = '0' ) then
        fifo_data_1(i)(31 downto 0) <= fifo_q_0(i)(31 downto 0);
        layer_1_state(i)(0) <= '1';
        if ( fifo_q_0(i)(63 downto 60) <= fifo_q_0(i + size1)(31 downto 28) and fifo_q_0(i)(63 downto 32) /= x"00000000" ) then
            fifo_data_1(i)(63 downto 32) <= fifo_q_0(i)(63 downto 32);
            layer_1_state(i)(1) <= '1';
            fifo_wen_1(i) <= '1';
            fifo_ren_0(i) <= '1';
        elsif ( fifo_q_0(i + size1)(63 downto 32) /= x"00000000" and fifo_empty_0(i + size1) = '0' and fifo_ren_0(i + size1) = '0' ) then
            fifo_data_1(i)(63 downto 32) <= fifo_q_0(i + size1)(31 downto 0);
            layer_1_state(i)(2) <= '1';
            fifo_wen_1(i) <= '1';
        end if;
    elsif ( fifo_empty_0(i + size1) = '0' and fifo_ren_0(i + size1) = '0' ) then
        fifo_data_1(i)(31 downto 0) <= fifo_q_0(i + size1)(31 downto 0);
        layer_1_state(i)(2) <= '1';
        if ( fifo_q_0(i)(31 downto 28) <= fifo_q_0(i + size1)(63 downto 60) and fifo_empty_0(i) = '0' and fifo_ren_0(i) = '0' ) then
            fifo_data_1(i)(63 downto 32) <= fifo_q_0(i)(31 downto 0);
            layer_1_state(i)(0) <= '1';
            fifo_wen_1(i) <= '1';
        elsif ( fifo_q_0(i + size1)(63 downto 32) /= x"00000000" ) then
            fifo_data_1(i)(63 downto 32) <= fifo_q_0(i + size1)(63 downto 32);
            layer_1_state(i)(3) <= '1';
            fifo_wen_1(i) <= '1';
            fifo_ren_0(i + size1) <= '1';
        end if;
    end if;
when "0011" =>
    layer_1_state(i) <= (others => '0');
when "1100" =>
    layer_1_state(i) <= (others => '0');
when "0101" =>
    if ( fifo_q_0(i)(63 downto 60) <= fifo_q_0(i + size1)(63 downto 60) and fifo_q_0(i)(63 downto 32) /= x"00000000" ) then
        fifo_data_1(i)(31 downto 0) <= fifo_q_0(i)(63 downto 32);
        layer_1_state(i)(0) <= '0';
        fifo_ren_0(i) <= '1';
    elsif ( fifo_q_0(i + size1)(63 downto 32) /= x"00000000" ) then
        fifo_data_1(i)(31 downto 0) <= fifo_q_0(i + size1)(63 downto 32);
        layer_1_state(i)(2) <= '0';
        fifo_ren_0(i + size1) <= '1';
    end if;
when "0100" =>
    -- TODO: define signal for empty since the fifo should be able to get empty if no hits are comming
    if ( fifo_empty_0(i) = '0' and fifo_ren_0(i) = '0' ) then
        -- TODO: what to do when fifo_q_0(i + size1)(63 downto 60) is zero? maybe error cnt?
        if ( fifo_q_0(i)(31 downto 28) <= fifo_q_0(i + size1)(63 downto 60) ) then
            fifo_data_1(i)(63 downto 32) <= fifo_q_0(i)(31 downto 0);
            layer_1_state(i)(0) <= '1';
            fifo_wen_1(i) <= '1';
        elsif ( fifo_q_0(i + size1)(63 downto 32) /= x"00000000" ) then
            fifo_data_1(i)(63 downto 32) <= fifo_q_0(i + size1)(63 downto 32);
            layer_1_state(i)(3) <= '1';
            fifo_wen_1(i) <= '1';
            fifo_ren_0(i + size1) <= '1';
        end if;
    else
        -- TODO: wait for fifo_0 i here --> error counter?
    end if;
when "0001" =>
    -- TODO: define signal for empty since the fifo should be able to get empty if no hits are comming
    if ( fifo_empty_0(i + size1) = '0' and fifo_ren_0(i + size1) = '0' ) then       
        -- TODO: what to do when fifo_q_0(i)(63 downto 60) is zero? maybe error cnt?     
        if ( fifo_q_0(i)(63 downto 60) <= fifo_q_0(i + size1)(31 downto 28) and fifo_q_0(i)(63 downto 32) /= x"00000000" ) then
            fifo_data_1(i)(63 downto 32) <= fifo_q_0(i)(63 downto 32);
            layer_1_state(i)(1) <= '1';
            fifo_wen_1(i) <= '1';
            fifo_ren_0(i) <= '1';
        else
            fifo_data_1(i)(63 downto 32) <= fifo_q_0(i + size1)(31 downto 0);
            layer_1_state(i)(2) <= '1';
            fifo_wen_1(i) <= '1';
        end if;
    else
        -- TODO: wait for fifo_0 i+size1 here --> error counter?
    end if;
when others =>
    layer_1_state(i) <= (others => '0');
"""
