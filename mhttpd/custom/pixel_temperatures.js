var watch_diff_cells = {}

var json_configuration = {}

var get_description = function (port, channel) {
    var descr = "Finding with port = " + port + ", channel = " + channel
    var found = false
    var node_js = json_configuration["Stations"]["Tracker"]["Direction"]
    for (direction in node_js) {
        for (layer in node_js[direction]["Layers"]) {
            for (ladder in node_js[direction]["Layers"][layer]["Ladders"]) {
                if (node_js[direction]["Layers"][layer]["Ladders"][ladder]["Temperature_parameters"]["port"] === port && node_js[direction]["Layers"][layer]["Ladders"][ladder]["Temperature_parameters"]["channel"] === channel) {
                    if (direction === "Upstream")
                        descr = "US, "
                    else
                        descr = "DS, "
                    descr += "Layer " + layer + ", Ladder " + ladder
                    if (found == true) {
                        descr += " : already found!"
                    }
                    found = true
                }
            }
        }
    }
    return descr
}


var init_table = function () {
    var table = document.getElementById("temp_tab")
    for (var ch = 0; ch < 48; ++ch) {
        if (ch == 46) {
            continue;
        }
        var row1 = table.insertRow(-1);
        if (ch % 8 == 0) {
            row1.style["border-top"] = "5px solid black";
        }
        var cell1 = row1.insertCell(0);
        var cell2 = row1.insertCell(1);
        var cell3 = row1.insertCell(2);
        var cell4 = row1.insertCell(3);
        var pos = from_input_to_portchannel(ch)
        cell1.innerHTML = "Input " + ch + ": port " + pos[0] + ", channel " + pos[1];
        descr = get_description(pos[0], pos[1])
        cell2.innerHTML = descr
        if (descr.search("Finding") != -1) {
            row1.style["background-color"] = "#666"
            cell2.innerHTML = "Not used"
        }
        cell3.classList.add("modbvalue")
        var link = "/Equipment/Pixel-SC-Temperature/Variables/Input[" + ch + "]"
        cell3.setAttribute("data-odb-path", link)
        cell3.id = "temp_ch_" + ch
        watch_diff_cells["temp_ch_" + ch] = cell4
        cell3.addEventListener('DOMSubtreeModified', function () {
            watch_diff_cells[this.id].innerHTML = convert_v_to_temperature(this.textContent);
        })
    }
}

var load_json = function () {
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function () {
        if (xmlhttp.status == 200 && xmlhttp.readyState == 4) {
            var words = xmlhttp.responseText;//.split(' ');
            json_configuration = JSON.parse(words);
            init_table()
        }
    }
    xmlhttp.open("GET", "mupix_configuration.json", true);
    xmlhttp.send();
}
load_json()