var lads = {}
var states = [
    "ladder_available",
    "ladder_power_on",
    "ladder_power_off"
]
var states_chip = [
    "chip_available",
    "chip_broken"
]

var current_selected = ""
var current_chip_selected = ""

var current_lad_chips = {}

var json_config = {}

var multiple_chip_boxes = {}

//var fileInput = document.querySelector('input[type="file"]');

function receivedText(e) {
    json_config = JSON.parse(e);
}

var load_json_config = function() {
    //var file = fileInput.files.item(0);
    var file = 'file:///C:/Users/Luigi/Programming/midas_pixel_page/mupix_configuration.json'
    var file2 = File("file:///C:/Users/Luigi/Programming/midas_pixel_page/mupix_configuration.json")
    var file3 = File.createFromFileName("mupix_configuration.json")
    var fr = new FileReader();
    fr.onload = function() {receivedText(fr.result)};
    //fr.readAsDataURL(file);
    fr.readAsText(file3)
}

function loadJSON(callback) {
    var xObj = new XMLHttpRequest();
    //xObj.overrideMimeType("application/json");
    //xObj.responseType = 'blob';
    xObj.open('GET', 'file:///C:/Users/Luigi/Programming/midas_pixel_page/mupix_configuration.json', true);
    xObj.onreadystatechange = function() {
    //xObj.onload = function() {
            if (xObj.readyState === 4 && xObj.status === 200) {
            // 2. call your callback function
            document.getElementById("debug").textContent = "OK: " + xObj.responseText
            callback(xObj.responseText);
        }
        else {
            document.getElementById("debug").textContent = "NO: " +JSON.stringify(xObj)
        }
    };
    xObj.send();
}

var init_config = function () {
    loadJSON(function(response) {
        json_config = response
        document.getElementById("debug").textContent = 'aaaa' + JSON.stringify(response)
    })
}

//json_config = document.getElementById("config").contentDocument
document.getElementById("debug").textContent = "AAAAA"
document.getElementById("debug").textContent = JSON.stringify(json_configuration)
json_config = json_configuration
//json_config = {}

//load_json_config()

//json_config = require("./mupix_configuration.json")

//init_config()

/*fetch('file:///C:/Users/Luigi/Programming/midas_pixel_page/mupix_configuration.json')
.then(response => {
    document.getElementById("debug").textContent = response
    document.getElementById("debug").textContent = JSON.stringify(response)
   return response.json();
})
.then(data => {json_config = data; document.getElementById("debug").textContent = data});
*/

var get_json_element = function (ladname) {
    var direction = ladname.substr(3,2)
    var layer = ladname.substr(1,1)
    var phi = ladname.substr(7,1)
    document.getElementById("debug").textContent = "---" + direction + "-" + layer + "-" + phi + "--" + JSON.stringify(json_config["Stations"]["Tracker"])
    if (direction === "US")
        return json_config["Stations"]["Tracker"]["Direction"]["Upstream"]["Layers"][layer]["Ladders"][phi]
    else if (direction === "DS")
        return json_config["Stations"]["Tracker"]["Direction"]["Downstream"]["Layers"][layer]["Ladders"][phi]
    else {
        return {"Direction" : "NotFound"}
    }
}

var create_ladder_element = function (lad) {
    var element = {
        "ladder" : lad,
        "state" : "available",
        "json_node" : get_json_element(lad.id)
    }
    return element
}

var create_chip_element = function (ladname, chip) {
    var element = {
        "chip" : chip,
        "state" : "available",
        "json_node" : lads[ladname]["json_node"]["Chips"][chip.id]
    }
    return element
}

var change_state = function(ladder, state) {
    class_state = "ladder_" + state
    if (state === "selected") {
        if ( ladder.classList.contains("ladder_unselected") )
            ladder.classList.remove("ladder_unselected")
        ladder.classList.add("ladder_selected")
    }
    else if (state === "unselected") {
        if ( ladder.classList.contains("ladder_selected") )
            ladder.classList.remove("ladder_selected")
        ladder.classList.add("ladder_unselected")
    }
    else {
        if (states.includes(class_state) == false) {
            return;
        }    
        for (var s = 0; s < states.length; s++) {
            if ( ladder.classList.contains(states[s]) )
                ladder.classList.remove(states[s])
        }
    ladder.classList.add(class_state)
    }
}

var change_state_chip = function(chip, state) {
    class_state = "ladder_" + state
    if (state === "selected") {
        if ( chip.classList.contains("chip_unselected") )
            chip.classList.remove("chip_unselected")
        chip.classList.add("chip_selected")
    }
    else if (state === "unselected") {
        if ( chip.classList.contains("chip_selected") )
            chip.classList.remove("chip_selected")
        chip.classList.add("chip_unselected")
    }
    else {
        if (states_chip.includes(class_state) == false) {
            return;
        }
        for (var s = 0; s < states_chip.length; s++) {
            if ( chip.classList.contains(states[s]) )
                chip.classList.remove(states[s])
        }
    chip.classList.add(class_state)
    }
}

var rotate_div = function(div, deg) {
	div.style.webkitTransform = 'rotate('+deg+'deg)'; 
    div.style.mozTransform    = 'rotate('+deg+'deg)'; 
    div.style.msTransform     = 'rotate('+deg+'deg)'; 
    div.style.oTransform      = 'rotate('+deg+'deg)'; 
    div.style.transform       = 'rotate('+deg+'deg)';
}

var reset_table = function (tabname) {
    var table2 = document.getElementById(tabname);
    table2.style["visibility"] = "hidden"
    var tableHeaderRowCount = 0;
    var rowCount = table2.rows.length;
    for (var i = tableHeaderRowCount; i < rowCount; i++) {
        table2.deleteRow(tableHeaderRowCount);
    }
}

var fill_table_by_json  = function (tablename, json_node, row_dict) {
    var table = document.getElementById(tablename);
    table.style["visibility"] = "visible"
    var row_n = 0
    document.getElementById("debug").textContent += JSON.stringify(row_dict)
    for (var key in row_dict) {
        var row1 = table.insertRow(row_n);
        var cell1 = row1.insertCell(0);
        var cell2 = row1.insertCell(1);
        cell1.innerHTML = key;
        document.getElementById("debug").textContent += json_node[row_dict[key]]
        cell2.innerHTML = json_node[row_dict[key]]
        row_n++
    }
}

var chip_clicked = function (ladname, chip) {
    document.getElementById("chip_table_header").style["visibility"] = "visible"
    document.getElementById("chip_dacs_header").style["visibility"] = "visible"
    document.getElementById("chip_dacs").style["visibility"] = "visible"
    document.getElementById("configure_chip_div").style["visibility"] = "visible"
    document.getElementById("pixel_configure_update").style["visibility"] = "hidden"

    var table = document.getElementById("pixel_parameters");
    table.style["visibility"] = "visible"
    document.getElementById("debug").textContent ="AAAA"+ chip + JSON.stringify(lads[ladname]["json_node"]["Chips"]) 
    document.getElementById("debug").textContent = JSON.stringify(current_lad_chips[chip]["json_node"])
    //for (cch = 0; cch < 3; ++cch) {
    //    if ( current_lad_chips[cch.toString()]["chip"].classList.contains("chip_selected") )
    //        current_lad_chips[cch.toString()]["chip"].classList.remove("chip_selected")
    //    change_state_chip(current_lad_chips[cch.toString()]["chip"], "unselected")
    //}
    if (current_chip_selected != "") {
        if (current_chip_selected !== chip)
            change_state_chip(current_lad_chips[current_chip_selected]["chip"], "unselected")
    }
    change_state_chip(current_lad_chips[chip]["chip"], "selected")
    current_chip_selected = chip
    reset_table("pixel_parameters")
    tab_fill = {"SpecBook ID:" : "SpecBookId", "MIDAS ID:" : "MIDAS_ID", "Event Display ID:" : "EventDisplayID", "Link A number:": "LinkA", "Link B number:": "LinkB", "Link C number:": "LinkC"}
    fill_table_by_json("pixel_parameters", current_lad_chips[chip]["json_node"], tab_fill)
    reset_table("Bias_tab")
    reset_table("Conf_tab")
    reset_table("VDAC_tab")
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
      tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
      tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
}

var ladder_clicked = function(ladname) {
    document.getElementById("ladder_header").style["visibility"] = "visible"
    lad = lads[ladname]["ladder"]
    document.getElementById("debug").textContent = lad.id + "-" + JSON.stringify(lads[ladname]["json_node"])
    change_state(lad, "selected")
    if (current_selected != "") {
        if (current_selected != ladname)
            change_state(lads[current_selected]["ladder"], "unselected")
    }
    current_selected = ladname
    reset_table("parameter_tab")
    tab_fill = {"SpecBook ID : " : "SpecBookId", "FEB ID : " : "FEB_ID", "DAB channel : " : "FEB_channel"}
    fill_table_by_json("parameter_tab", lads[ladname]["json_node"], tab_fill)

    reset_table("ladder_lv_tab")
    tab_fill = {"Hameg Module : " : "Module", "IP address : " : "IP", "Hameg Channel : " : "Channel"}
    fill_table_by_json("ladder_lv_tab", lads[ladname]["json_node"]["LV_parameters"], tab_fill)

    document.getElementById("ladder_lv_header").style["visibility"] = "visible"
    document.getElementById("ladder_lv_settings").style["visibility"] = "visible"

    document.getElementById("ladder_lv_read_volt").setAttribute("data-odb-path", "/Equipment/HAMEG" + lads[ladname]["json_node"]["LV_parameters"]["Module"].toString()+ "/Variables/Voltage[" + lads[ladname]["json_node"]["LV_parameters"]["Channel"].toString()  + "]")
    document.getElementById("ladder_lv_read_curr").setAttribute("data-odb-path", "/Equipment/HAMEG" + lads[ladname]["json_node"]["LV_parameters"]["Module"].toString()+ "/Variables/Current[" + lads[ladname]["json_node"]["LV_parameters"]["Channel"].toString()  + "]")

    reset_table("ladder_hv_tab")
    tab_fill = {"HV-box Module : " : "Module", "IP address : " : "IP", "HV-box Channel : " : "Channel"}
    fill_table_by_json("ladder_hv_tab", lads[ladname]["json_node"]["HV_parameters"], tab_fill)

    document.getElementById("ladder_hv_header").style["visibility"] = "visible"
    document.getElementById("ladder_hv_settings").innerHTML = ""

    document.getElementById("chip_header").style["visibility"] = "visible"
    document.getElementById("pixel_select").style["visibility"] = "visible"
    document.getElementById("pixel_select").innerHTML = ""
    ladder_up = document.createElement("div")
    ladder_up.classList.add("chip_LadderPortionUp")
    var direction = ladname.substr(3,2)
    if (direction == "US")
        document.getElementById("pixel_select").appendChild(ladder_up)
    current_lad_chips = {}
    for (var ch = 0; ch < 3; ++ ch) {
        let chip = document.createElement("div")
        chip.type = "submit"
        chip.classList.add("chipDiv")
        chip.classList.add("chip_available")
        chip.classList.add("chip_unselected")
        chip.textContent = "Chip " + ch.toString()
        let ch_num = ch.toString()
        chip.id = ch_num
        chip.onclick = function () {chip_clicked(ladname, ch_num)}
        document.getElementById("pixel_select").appendChild(chip)
        current_lad_chips[chip.id] = create_chip_element(ladname, chip)
    }
    if (direction == "DS")
        document.getElementById("pixel_select").appendChild(ladder_up)
    ladder_down = document.createElement("div")
    ladder_down.classList.add("chip_LadderPortionDown")
    document.getElementById("pixel_select").appendChild(ladder_down)
    document.getElementById("chip_table_header").style["visibility"] = "hidden"
    document.getElementById("chip_dacs_header").style["visibility"] = "hidden"
    document.getElementById("chip_dacs").style["visibility"] = "hidden"
    document.getElementById("configure_chip_div").style["visibility"] = "hidden"
    document.getElementById("pixel_configure_update").style["visibility"] = "hidden"
    reset_table("pixel_parameters")
    reset_table("Bias_tab")
    reset_table("Conf_tab")
    reset_table("VDAC_tab")
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
      tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
      tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
}

function openChipDacs(evt, dacName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
      tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
      tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    if (dacName == "Hide") {
        return;
    }
    document.getElementById(dacName).style.display = "block";

    var odb_dacName = ""
    if (dacName === "Bias")
        odb_dacName = "BIASDACS"
    else if (dacName === "Conf")
        odb_dacName = "CONFDACS"
    else if (dacName === "VDAC")
        odb_dacName = "VDACS"
    document.getElementById(dacName + "_head").textContent = "Change values in " + "/Equipment/Mupix/Settings/" + odb_dacName + "/" + (parseInt(lads[current_selected]["json_node"]["FEB_ID"])*12 + parseInt(current_lad_chips[current_chip_selected]["json_node"]["MIDAS_ID"]))
    reset_table(dacName + "_tab")
    var table = document.getElementById(dacName + "_tab");
    table.style["visibility"] = "visible"
    dac_list = mupix_dacs[dacName]
    document.getElementById("debug").textContent = JSON.stringify(dac_list)
    var row_n = 0
    for (dac in dac_list) {
        var row1 = table.insertRow(row_n);
        var cell1 = row1.insertCell(0);
        var cell2 = row1.insertCell(1);
        cell1.innerHTML = dac_list[dac];
        cell2.classList.add("modbvalue")
        var link = "/Equipment/Mupix/Settings/" + odb_dacName + "/" + (parseInt(lads[current_selected]["json_node"]["FEB_ID"])*12 + parseInt(current_lad_chips[current_chip_selected]["json_node"]["MIDAS_ID"])) + "/" + dac_list[dac]
        cell2.setAttribute("data-odb-path", link)
        cell2.setAttribute("data-odb-editable", "1")
        row_n++
    }
    evt.currentTarget.className += " active";
}

function configureChip(evt) {
    var mask_names = []
    var px_mask = []
    var num = (parseInt(lads[current_selected]["json_node"]["FEB_ID"])*12 + parseInt(current_lad_chips[current_chip_selected]["json_node"]["MIDAS_ID"]))
    for (var i = 0; i < 128; ++i) {
        mask_names.push("/Equipment/Mupix/Settings/Daq/mask[" + i + "]")
        if (i === num)
            px_mask.push(false);
        else
            px_mask.push(true)
    }
    document.getElementById("pixel_configure_update").style["visibility"] = "visible"
    document.getElementById("pixel_configure_update").textContent = "Configuring chip number " + num
    mjsonrpc_db_paste(mask_names, px_mask).then(function () {
        mjsonrpc_db_paste(["/Equipment/Switching/Settings/MupixConfig"], [true])
    }).catch(function(error) {
        mjsonrpc_error_alert(error);
     });
}

function configureMultipleChip(evt) {

    var mask_names = []
    var px_mask = []
    document.getElementById("debug").textContent  = "Configuring : "
    for (direction in multiple_chip_boxes) {
        for (layer in multiple_chip_boxes[direction]) {
            for (ladder in multiple_chip_boxes[direction][layer]) {
                for (chip in multiple_chip_boxes[direction][layer][ladder]) {
                    var lad_name = "L" + layer
                    if (direction == "Upstream")
                        lad_name += "_US"
                    else if (direction == "Downstream")
                        lad_name += "_DS"
                    else
                        return
                    lad_name += "_L" +ladder
                    num = (parseInt(lads[lad_name]["json_node"]["FEB_ID"])*12 + parseInt(lads[lad_name]["json_node"]["Chips"][chip]["MIDAS_ID"]))
                    mask_names.push("/Equipment/Mupix/Settings/Daq/mask[" + num + "]")
                    if (multiple_chip_boxes[direction][layer][ladder][chip].checked == true) {
                        document.getElementById("debug").textContent += lad_name + " : " + num + " - "
                        px_mask.push(false);
                    }
                    else
                        px_mask.push(true);
                }
            }
        }
    }

    mjsonrpc_db_paste(mask_names, px_mask).then(function () {
        mjsonrpc_db_paste(["/Equipment/Switching/Settings/MupixConfig"], [true])
    }).catch(function(error) {
        mjsonrpc_error_alert(error);
     });
}

var create_octagon = function(ele, prefix, inversion) {
    for (var i = 0; i < 8; ++i) {
        //lad = document.getElementById("L0_DS").createElement("div")
        let lad = document.createElement("div")
        lad.type = "submit"
        lad.classList.add("ladderDiv")
        lad.classList.add("ladder_available")
        lad.classList.add("ladder_unselected")
        //lad.value = "Ladder " + i.toString()
        lad.id = prefix + i.toString()
        lad.style = "width:70px;height:20px;position: relative;"
        lad.textContent = "Ladder " + i.toString()
        if (inversion == false) {
            leftpos = 120*Math.cos(Math.PI/8 + (2*Math.PI*(i-4)/8)) +120
            toppos = 120*(Math.sin(Math.PI/8 + (2*Math.PI*(i-4)/8))) -20*i + 130
            lad.style.left = leftpos.toString() + "px"
            lad.style.top = toppos.toString() + "px"
            rotate_div(lad, -90 + 22.5 + (360*i/8))
        }
        else {
            leftpos = 120*Math.cos(-Math.PI/8-(2*Math.PI*(i)/8)) +120
            toppos = 120*(Math.sin(-Math.PI/8-(2*Math.PI*(i)/8))) -20*i + 130
            lad.style.left = leftpos.toString() + "px"
            lad.style.top = toppos.toString() + "px"
            rotate_div(lad, 90 - 22.5 - (360*i/8))
        }
        /*if(lad.addEventListener) {
            lad.addEventListener("click", ladder_clicked(prefix + i.toString()));
        }
        else { 
            lad.attachEvent("onclick", ladder_clicked(prefix + i.toString()))
        }*/
        lad.onclick = function () {ladder_clicked(lad.id)}
        ele.appendChild(lad)
        lads[prefix + i.toString()] = create_ladder_element(lad)
    }
}

var create_decagon = function(ele, prefix, inversion) {
    for (var i = 0; i < 10; ++i) {
        //lad = document.getElementById("L0_DS").createElement("div")
        let lad = document.createElement("div")
        lad.type = "submit"
        lad.classList.add("ladderDiv")
        lad.classList.add("ladder_available")
        lad.classList.add("ladder_unselected")
        //lad.value = "Ladder " + i.toString()
        lad.id = prefix + i.toString()
        //lad.width = "70px"
        lad.style = "width:70px;height:20px;position: relative;"
        lad.textContent = "Ladder " + i.toString()
        if (inversion == false) {
            leftpos = 120*Math.cos(Math.PI/10 + (2*Math.PI*(i-5)/10)) +120
            toppos = 120*(Math.sin(Math.PI/10 + (2*Math.PI*(i-5)/10))) -20*i + 130
            lad.style.left = leftpos.toString() + "px"
            lad.style.top = toppos.toString() + "px"
            rotate_div(lad, -72 + (360*i/10))
        }
        else {
            leftpos = 120*Math.cos(-Math.PI/10 -(2*Math.PI*(i)/10)) +120
            toppos = 120*(Math.sin(-Math.PI/10 -(2*Math.PI*(i)/10))) -20*i + 130
            lad.style.left = leftpos.toString() + "px"
            lad.style.top = toppos.toString() + "px"
            rotate_div(lad, 72 - (360*i/10))
        }
        /*if(lad.addEventListener) {
            lad.addEventListener("click", ladder_clicked(prefix + i.toString()));
        }
        else { 
            lad.attachEvent("onclick", ladder_clicked(prefix + i.toString()))
        }*/
        lad.onclick = function () {ladder_clicked(lad.id)}
        ele.appendChild(lad)
        //TODO: nest it like the rest
        lads[prefix + i.toString()] = create_ladder_element(lad)
    }
}

var list_chips_configuration = function (ele, direction) {
    var node_us = json_config["Stations"]["Tracker"]["Direction"][direction]
    multiple_chip_boxes[direction] = {}
    for (layer in node_us["Layers"]) {
        multiple_chip_boxes[direction][layer] = {}
        for (ladder in node_us["Layers"][layer]["Ladders"]) {
            multiple_chip_boxes[direction][layer][ladder] = {}
            for (chip in node_us["Layers"][layer]["Ladders"][ladder]["Chips"]) {
                var chip_cb_div = document.createElement("div")
                var chip_line = document.createElement("input")
                chip_line.type = "checkbox"
                chip_line.id = direction + "_L" + layer + "_L" + ladder + "_C" + chip
                chip_line.name = direction + "_L" + layer + "_L" + ladder + "_C" + chip
                var chip_label = document.createElement("label")
                chip_label.setAttribute("for", direction + "_L" + layer + "_L" + ladder + "_C" + chip)
                chip_label.textContent = "Layer " + layer + " - Ladder " + ladder + " - Chip " + chip
                multiple_chip_boxes[direction][layer][ladder][chip] = chip_line
                chip_cb_div.appendChild(chip_line)
                chip_cb_div.appendChild(chip_label)
                ele.appendChild(chip_cb_div)
            }
        }
    }
}

create_octagon(document.getElementById("L0_US"), "L0_US_L", false)
create_decagon(document.getElementById("L1_US"), "L1_US_L", false)
create_octagon(document.getElementById("L0_DS"), "L0_DS_L", true)
create_decagon(document.getElementById("L1_DS"), "L1_DS_L", true)

list_chips_configuration(document.getElementById("multiple_chip_configuration_us"), "Upstream")
list_chips_configuration(document.getElementById("multiple_chip_configuration_ds"), "Downstream")
