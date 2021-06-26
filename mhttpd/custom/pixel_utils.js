var from_portchannel_to_input = function (port, channel) {
    return port * 8 + channel;//Is it right??
}

var from_input_to_portchannel = function (inp) {
    port = parseInt(inp / 8)
    channel = inp % 8 //Is it right?
    return [port, channel]
}

var convert_v_to_temperature = function (volt) {
    var I_0 = 950;
    var a_0 = 2;
    var I_1 = 1300;
    var a_1 = 2;
    var p0_0 = 228;
    var p1_0 = -0.3318;
    var p1_1 = p1_0 * (I_1 / I_0) * (a_0 / a_1);
    return p0_0 + p1_0 * (volt*1000);
}