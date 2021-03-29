 
class Node:
    def __init__(self, val, val_l, function, ifelse, layer, cnt_if, cnt_else, left=None, right=None, last_direction=None):
        self.val = val
        self.val_l = val_l
        self.function = function
        self.ifelse = ifelse
        self.layer = layer
        self.cnt_if = cnt_if
        self.cnt_else = cnt_else
        self.left = left
        self.right = right
        self.last_direction = last_direction
    
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

d_dict = {'Right': 'else', 'Left': 'if', None: None}
def setTreeOutput(node, level=0):
    if node != None:
        if node.function == 'if':
            row = 1
        if level > 0:
            output.append([level, node.function, node.left, node.val_l, node.ifelse, node.layer, node.cnt_if, node.cnt_else, node.val, node.last_direction])
        setTreeOutput(node.left, level + 1)
        setTreeOutput(node.right, level + 1)

def printOutput():
    for val in output:
        t = ''.join(['\t' for i in range(val[0])])
        if val[4] == 'if': 
            if val[6] == 0: 
                state_val = 0
            else:
                state_val = val[6]
        if val[4] == 'else': 
            if val[7] == 0:
                state_val = layer_dict[val[5]]//2
            else:
                state_val = layer_dict[val[5]]//2 + val[7]
        
        if val[8] > 1 and state_val < val[8]-1 and val[7] == 0: 
            state_val = val[6] + 1
            
        if d_dict[val[9]] != val[4] and val[9] != None and val[8] <= 1:
            state_val -= 1
        if d_dict[val[9]] == 'else' and val[4] == 'if':
            state_val -= 1
            
        if val[6] == 0 and val[7] == 1 and val[0] == 3 and val[4] == 'if' and d_dict[val[9]] == 'if':
            state_val += 1
            
        if state_val == -1: state_val = 0
        
        if val[4] == 'else' and d_dict[val[9]] == 'else':
            state_val += 1
            
        state = write_state(val[3][0], state_val)
        debug = ''
        debug = str(val[6]) + str(val[7]) + str(val[0]) + str(layer_dict[val[5]]//2) + str(state_val) + val[4] + str(d_dict[val[9]])
        if val[2] == None:
            print('{}{}{}\n{}  {}\n{}  last'.format(t, val[1], debug, t, state, t))
        elif val[2] == 'None':
            continue
        else:
            print('{}{}{}\n{}  {}'.format(t, val[1], debug, t, state))

def generate(cnt, stop=3, direction=None, last_direction=None, layer=0, cnt_if=0, cnt_else=0):
    cur_last_direction = None
    if direction == 'Left':
        if last_direction == 'Left':
            last = 1
        else:
            last = 0
        if last_direction == 'Left':
            if (cnt_else - 1) >= 0:
                last_right = -1
            else:
                last_right = 0
        else:
            last_right = 0
        if last_direction == 'Right':
            last_right = 1
        else:
            last_right = 0
            
        val = write_if(layer, 31 + (cnt_if+last)*32, 31 + (cnt_else+last_right)*32)
        root = Node(cnt, [layer, 31 + (cnt-1)*32, 31], val, 'if', layer, cnt_if, cnt_else, 'if', 'else', last_direction)
        cur_last_direction = 'Left'
    elif direction == 'Right':
        if last_direction == 'Right':
            last = 1
        else:
            last = 0
            
        val = write_elsif(layer, 31 + (cnt_else+last)*32, 31)
        root = Node(cnt, [layer, 31 + (cnt-1)*32, 31], val, 'else', layer, cnt_if, cnt_else, 'if', 'else', last_direction)
        cur_last_direction = 'Right'
    else:
        root = Node(0, 0, 'None', None, 0, 0, 0, 'None', 'None')
    
    if last_direction == 'Left': cnt_if += 1
    if last_direction == 'Right': cnt_else += 1
        
    if cnt == stop:
        if direction == 'Left':
            if last_direction == 'Left':
                if (cnt_else - 1) >= 0:
                    last_right = 0#-1
                else:
                    last_right = 0
            else:
                last_right = 0
                #21442
            #if last_direction == 'Right':
                #last_right = 1
            #else:
                #last_right = 0
                
            val = write_if(layer, 31 + (cnt_if)*32, 31 + (cnt_else+last_right)*32)
            root = Node(cnt, [layer, 31 + (cnt-1)*32, 31], val, 'if', layer, cnt_if, cnt_else, None, None)
            cnt_if += 1
        else:
            if last_direction == 'Left':
                if (cnt_else - 1) >= 0:
                    last_right = -1
                else:
                    last_right = 0
            else:
                last_right = 0
                
            val = write_elsif(layer, 31 + (cnt_else)*32, 31 + (cnt_else+last_right)*32)
            root = Node(cnt, [layer, 31 + (cnt-1)*32, 31], val, 'else', layer, cnt_if, cnt_else,  None, None)
            cnt_else += 1
        return root
    cnt += 1
    root.left = generate(cnt, stop, 'Left', cur_last_direction, layer, cnt_if, cnt_else)
    root.right = generate(cnt, stop, 'Right', cur_last_direction, layer, cnt_if, cnt_else)

    return root

layer_dict = {0:4, 1:8, 2:16, 3:32}

for i, layer in enumerate([4, 8]):
    #if i == 0: continue
    out_string = ""
    out_string += "when \"{}\" =>".format("".join(["0" for i in range(layer)]))
    out_string += "\n"

    output = []
    root = generate(0, stop=layer/2, layer=i)

    setTreeOutput(root)
    printOutput()
 