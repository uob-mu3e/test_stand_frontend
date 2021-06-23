var from_portchannel_to_input = function (port, channel) {
    return port * 8 + channel;//Is it right??
}

var from_input_to_portchannel = function (inp) {
    port = parseInt(inp / 8)
    channel = inp % 8 //Is it right?
    return [port, channel]
}

var convert_v_to_temperature = function (volt) {
    return 2 * volt;
}