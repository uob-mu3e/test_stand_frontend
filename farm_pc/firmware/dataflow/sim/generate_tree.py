
class Node:
    def __init__(self, val, val_l, function, ifelse, layer, left=None, right=None):
        self.val = val
        self.val_l = val_l
        self.function = function
        self.ifelse = ifelse
        self.layer = layer
        self.left = left
        self.right = right
    
    def __str__(self):
        return str(self.val) + ' ' + str(self.function)

def write_if(fifo_layer, range_0, range_1):
    format_l = [fifo_layer, range_0, range_0-3, fifo_layer, \
                fifo_layer, range_1, range_1-3, fifo_layer, range_0, \
                range_0-31, fifo_layer, fifo_layer]
    out = "if ( fifo_q_{}(i)({} downto {}) <= fifo_q_{}(i + size{})({} downto {})".format(*format_l[:7])
    out += " and fifo_q_{}(i)({} downto {}) /= x\"00000000\" and fifo_empty_{}(i) = ".format(*format_l[7:11])
    out += "'0' and fifo_ren_{}(i) = '0' ) then".format(format_l[-1])
    return out

def write_elsif(fifo_layer, range_0, range_1):
    format_l = [fifo_layer, fifo_layer+1, range_0, range_0-31, fifo_layer, fifo_layer+1, fifo_layer, fifo_layer+1]
    out = "elsif ( fifo_q_{}(i + size{})({} downto {}) /= x\"00000000\" and".format(*format_l[:4])
    out +=" fifo_empty_{}(i + size{}) = '0' and fifo_ren_{}(i + size{}) = '0' ) then".format(*format_l[4:8])
    return out

def write_data(fifo_layer, range_0, range_1):
    format_l = [fifo_layer+1, range_0, range_0-31, fifo_layer, range_1, range_1-31]
    return "fifo_data_{}(i)({} downto {}) <= fifo_q_{}(i)({} downto {});".format(*format_l)

def write_state(fifo_layer, state):
    format_l = [fifo_layer+1, state]
    return "layer_{}_state(i)({}) <= '1';".format(*format_l)

def write_wen(fifo_layer):
    format_l = [fifo_layer+1]
    return "fifo_wen_{}(i) <= '1';".format(*format_l)

def write_ren(fifo_layer, size=False):
    if size:
        format_l = [fifo_layer, fifo_layer+1]
        return "fifo_ren_{}(i + size{}) <= '1';".format(*format_l)
    else:
        format_l = [fifo_layer]
        return "fifo_ren_{}(i) <= '1';".format(*format_l)

def setTreeOutput(node, level=0):
    if node != None:
        if node.function == 'if':
            row = 1
        if level > 0:
            output.append([level, node.function, node.left, node.val_l, node.ifelse, node.layer])
        setTreeOutput(node.left, level + 1)
        setTreeOutput(node.right, level + 1)

def printOutput():
    # val = [level, node.function, node.left, node.val_l, node.ifelse, node.layer]
    for val in output:
        t = ''.join(['\t' for i in range(val[0])])
        if val[4] == 'if': state_val = (val[0] - 1)
        if val[4] == 'else': state_val = (val[0] - 2) + 2 ** (val[5]+1)


        state = write_state(val[3][0], state_val)
        if val[2] == None:
            print('{}{}\n{}  {}\n{}  last'.format(t, val[1], t, state, t))
        else:
            print('{}{}\n{}  {}'.format(t, val[1], t, state))

def generate(cnt, stop=3, direction=None, layer=0):
    if direction == 'Left':
        val = write_if(layer, 31 + (cnt-1)*32, 31)
        root = Node(cnt, [layer, 31 + (cnt-1)*32, 31], val, 'if', layer, 'if', 'else')
    else:
        root = Node(cnt, [layer, 31 + (cnt-1)*32, 31], 'elif', 'else', layer, 'if', 'else')
    if cnt == stop:
        if direction == 'Left':
            val = write_if(layer, 31 + (cnt-1)*32, 31)
            root = Node(cnt, [layer, 31 + (cnt-1)*32, 31], val, 'if', layer, None, None)
        else:
            root = Node(cnt, [layer, 31 + (cnt-1)*32, 31], 'elif', 'else', layer, None, None)
        return root
    cnt += 1
    root.left = generate(cnt, stop, 'Left', layer)
    root.right = generate(cnt, stop, 'Right', layer)

    return root



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




for i, layer in enumerate([8]):
    out_string = ""
    out_string += "when \"{}\" =>".format("".join(["0" for i in range(layer)]))
    out_string += "\n"

    output = []
    root = generate(0, stop=layer/2, direction='Left', layer=i)

    setTreeOutput(root)
    printOutput()
 