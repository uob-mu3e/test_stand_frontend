
def write_tabs(n_tabs):
    return "".join("\t" for i in range(n_tabs))

def write_if(n_tabs, fifo_layer, range_0, range_1, invert):
    if invert:
        return write_elsif(n_tabs, fifo_layer, range_0, range_1)
    format_l = [write_tabs(n_tabs), fifo_layer, range_0, range_0-3, fifo_layer, \
                fifo_layer, range_1, range_1-3, fifo_layer, range_0, \
                range_0-31, fifo_layer, fifo_layer]
    out = "{}if ( fifo_q_{}(i)({} downto {}) <= fifo_q_{}(i + size{})({} downto {})".format(*format_l[:8])
    out += " and fifo_q_{}(i)({} downto {}) /= x\"00000000\" and fifo_empty_{}(i) = ".format(*format_l[8:12])
    out += "'0' and fifo_ren_{}(i) = '0' ) then".format(format_l[-1])
    return out

def write_elsif(n_tabs, fifo_layer, range_0, range_1):
    format_l = [write_tabs(n_tabs), fifo_layer, fifo_layer+1, range_0, range_0-31, fifo_layer, fifo_layer+1, fifo_layer, fifo_layer+1]
    out = "{}elsif ( fifo_q_{}(i + size{})({} downto {}) /= x\"00000000\" and".format(*format_l[:5])
    out +=" fifo_empty_{}(i + size{}) = '0' and fifo_ren_{}(i + size{}) = '0' ) then".format(*format_l[5:9])
    return out

def write_data(n_tabs, fifo_layer, range_0, range_1):
    format_l = [write_tabs(n_tabs), fifo_layer+1, range_0, range_0-31, fifo_layer, range_1, range_1-31]
    return "{}fifo_data_{}(i)({} downto {}) <= fifo_q_{}(i)({} downto {});".format(*format_l)

def write_state(n_tabs, fifo_layer, state):
    format_l = [write_tabs(n_tabs), fifo_layer+1, state]
    return "{}layer_{}_state(i)({}) <= '1';".format(*format_l)

def write_wen(n_tabs, fifo_layer):
    format_l = [write_tabs(n_tabs), fifo_layer+1]
    return "{}fifo_wen_{}(i) <= '1';".format(*format_l)

def write_ren(n_tabs, fifo_layer, size=False):
    if size:
        format_l = [write_tabs(n_tabs), fifo_layer, fifo_layer+1]
        return "{}fifo_ren_{}(i + size{}) <= '1';".format(*format_l)
    else:
        format_l = [write_tabs(n_tabs), fifo_layer]
        return "{}fifo_ren_{}(i) <= '1';".format(*format_l)

def first_leaf(out_string, layer, div, k, rounds, fifo_layer, invert=False):
    
    if invert:
        loop = [i for i in reversed(range(int(layer/div)))]
    else:
        loop = [i for i in range(int(layer/div))]

    for j in loop:
        out_string += write_if(n_tabs=1+j+rounds*int(layer/2), fifo_layer=fifo_layer, range_0=31+32*(j+rounds*int(layer/4)), range_1=31, invert=invert)
        out_string += "\n"
        out_string += write_data(n_tabs=2+j+rounds*int(layer/2), fifo_layer=fifo_layer, range_0=31+32*(j+rounds*int(layer/4)), range_1=31+32*j)
        out_string += "\n"
        out_string += write_state(n_tabs=2+j+rounds*int(layer/2), fifo_layer=fifo_layer, state=j+rounds*int(layer/4))
        out_string += "\n"
        if j+1 == layer/2:
            out_string += write_wen(n_tabs=2+int(layer/div) - 1, fifo_layer=i)
            out_string += "\n"
            if not invert:
                out_string += write_ren(n_tabs=2+int(layer/div) - 1, fifo_layer=i, size=((k+1)%2==0))
                out_string += "\n"
    rounds += int(layer/div) - 1

    if layer/div == layer/2:
        return out_string
    else:
        return first_leaf(out_string, layer, div*2, k, rounds, fifo_layer, invert)




for i, layer in enumerate([4, 8]):
    out_string = ""
    out_string += "when \"{}\" =>".format("".join(["0" for i in range(layer)]))
    # out_string += "\n"
    # out_string += write_if(n_tabs=1, fifo_layer=i, range_0=31, range_1=31)
    # out_string += "\n"
    # out_string += write_data(n_tabs=2, fifo_layer=i, range_0=31, range_1=31)
    # out_string += "\n"
    # out_string += write_state(n_tabs=2, fifo_layer=i, state=0)
    out_string += "\n"

    for k in range(2):
        cur_string = ""
        out_string += first_leaf(cur_string, layer, 2, k, 0, i, ((k+1)%2==0))
        
        # out_string += "AAAAAAAAAAAAA"
        # out_string += "\n"
    print(out_string)

    # break
        # for j in range(int(layer/4)):
        #     out_string += write_if(n_tabs=2+j, fifo_layer=i, range_0=63+32*j, range_1=31)
        #     out_string += "\n"
        #     out_string += write_data(n_tabs=3+j, fifo_layer=i, range_0=63+32*j, range_1=63+32*j)
        #     out_string += "\n"
        #     out_string += write_state(n_tabs=3+j, fifo_layer=i, state=j+1)
        #     out_string += "\n"


        # out_string += write_wen(n_tabs=3+j, fifo_layer=i)
        # out_string += "\n"
        # out_string += write_ren(n_tabs=3+j, fifo_layer=i, size=((k+1)%2==0))
        # out_string += "\n"

            
            # range_0 = (64 + 32*j)-1
            # tabs = "".join("\t" for i in range(3 + j))
            # range_01 = (32 + 32*j)
            # formate_l2 = [tabs, i+1, range_0, range_01, i, range_0, range_01]
            # out_string += "{}fifo_data_{}(i)({} downto {}) <= fifo_q_{}(i)({} downto {});".format(*formate_l2)
            # out_string += "\n"
            # out_string += "{}layer_{}_state(i)({}) <= '1';".format(tabs, i+1, j+1)
            # out_string += "\n"

            # tabs = "".join("\t" for i in range(3 + j))
            # out_string += "{}fifo_wen_{}(i) <= '1';".format(tabs, i+1)
            # out_string += "\n"
            # out_string += "{}fifo_ren_{}(i) <= '1';".format(tabs, i)
            # out_string += "\n"

            # tabs = "".join("\t" for i in range(2 + j))
            # range_0 = (64 + 32*j)-1
            # range_01 = (32 + 32*j)
            # format_l = [tabs, i, i+1, range_0, range_01, i, i+1, i, i+1]
            # out_string += "{}elsif ( fifo_q_{}(i + size{})({} downto {}) /= x\"00000000\" and fifo_empty_{}(i + size{}) = '0' and fifo_ren_{}(i + size{}) = '0' ) then;".format(*format_l)
            # out_string += "\n"

            # out_string += "fifo_data_1(i)(63 downto 32) <= fifo_q_0(i + size1)(31 downto 0);"
            # out_string += "\n"
            # out_string += "layer_1_state(i)(2) <= '1';"
            # out_string += "\n"
            # out_string += "fifo_wen_1(i) <= '1';"
            # out_string += "\n"

        # out_string += "elsif ( fifo_empty_0(i + size1) = '0' and fifo_ren_0(i + size1) = '0' ) then"
        # out_string += "\n"
        # out_string += "fifo_data_1(i)(31 downto 0) <= fifo_q_0(i + size1)(31 downto 0);"
        # out_string += "\n"
        # out_string += "layer_1_state(i)(2) <= '1';"
        # out_string += "\n"

        # for j in range(int(layer/4)):
        #     tabs = "".join("\t" for i in range(2 + j))
        #     range_0 = (64 + 32*j)-1
        #     range_1 = (32)-1
        #     formate_l = [tabs, i, range_0, range_0-3, i, i, range_1, range_1-3, i, range_0, range_0-3]
        #     out_string += "if ( fifo_q_0(i)(31 downto 28) <= fifo_q_0(i + size1)(63 downto 60) and fifo_empty_0(i) = '0' and fifo_ren_0(i) = '0' ) then"
        #     out_string += "\n"

        #     out_string += "fifo_data_1(i)(63 downto 32) <= fifo_q_0(i)(31 downto 0);"
        #     out_string += "\n"
        #     out_string += "layer_1_state(i)(0) <= '1';"
        #     out_string += "\n"


        #     out_string += "fifo_wen_1(i) <= '1';"
        #     out_string += "\n"




    # # break
    # print(out_string)

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
