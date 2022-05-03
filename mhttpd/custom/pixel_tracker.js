var lads = {}
var states = [
    "ladder_available",
    "ladder_power_on",
    "ladder_power_off",
    "ladder_power_current_high",
    "ladder_hv_current_high",
    "ladder_hv_on"
]
var states_chip = [
    "chip_available",
    "chip_broken"
]

var current_selected = ""
var current_chip_selected = ""

var current_lad_chips = {}

var json_config = {}
var mupix_dacs = {}

var multiple_chip_boxes = {}

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
        "state" : "available"
        //"json_node" : get_json_element(lad.id)
    }
    if (lad.hasAttribute("id"))
        element["json_node"] = get_json_element(lad.id)
    return element
}

var create_chip_element = function (ladname, chip) {
    var element = {
        "chip" : chip,
        "state" : ["available"],
        "json_node" : lads[ladname]["json_node"]["Chips"][chip.id]
    }
    return element
}

var change_state = function (ladder, state, add = false) {
    lad_div = ladder["ladder"]
    class_state = "ladder_" + state
    if (state === "selected") {
        if (lad_div.classList.contains("ladder_unselected") )
            lad_div.classList.remove("ladder_unselected")
        lad_div.classList.add("ladder_selected")
    }
    else if (state === "unselected") {
        if (lad_div.classList.contains("ladder_selected") )
            lad_div.classList.remove("ladder_selected")
        lad_div.classList.add("ladder_unselected")
    }
    else {
        if (states.includes(class_state) == false) {
            return;
        }
        if (add == false) {
            document.getElementById("debug").textContent = "Ladder " + lad_div.id + " set not to blink"
            for (var s = 0; s < states.length; s++) {
                if (lad_div.classList.contains(states[s]))
                    lad_div.classList.remove(states[s])
            }
            ladder["state"] = [state]
            lad_div.classList.add(class_state)
        }
        else {
            document.getElementById("debug").textContent = "Ladder " + lad_div.id + " set to blink"
            ladder["state"].push(state)
            ladder["blinking"] = 0
            setInterval(function () {
                if (ladder["blinking"] >= ladder["state"].length)
                    ladder["blinking"] = 0
                class_state_blink = "ladder_" + ladder["state"][ladder["blinking"]]
                document.getElementById("debug").textContent = "Ladder " + lad_div.id + " blinking at " + ladder["blinking"]
                for (var s = 0; s < states.length; s++) {
                    if (lad_div.classList.contains(states[s]))
                        lad_div.classList.remove(states[s])
                }
                lad_div.classList.add(class_state_blink)
            }, 1000);
        }
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

var reset_table = function (tabname, tableHeaderRowCount) {
    var table2 = document.getElementById(tabname);
    table2.style["visibility"] = "hidden"
    //var tableHeaderRowCount = 0;
    var rowCount = table2.rows.length;
    for (var i = tableHeaderRowCount; i < rowCount; i++) {
        table2.deleteRow(tableHeaderRowCount);
    }
}

var fill_table_by_json  = function (tablename, json_node, row_dict) {
    var table = document.getElementById(tablename);
    table.style["visibility"] = "visible"
    var row_n = 0
    if (table.hasAttribute("rows") === true)
        row_n = table.rows.length;
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
    reset_table("pixel_parameters", 0)
    tab_fill = {"SpecBook ID:" : "SpecBookId", "Configuration ID:" : "MIDAS_ID", "Event Display ID:" : "EventDisplayID", "Link A number:": "LinkA", "Link B number:": "LinkB", "Link C number:": "LinkC", "Status" : "Initial_Status", "Optimized VPVCO" : "opt_VPVCO", "Optimized VNVCO" : "opt_VNVCO"}
    fill_table_by_json("pixel_parameters", current_lad_chips[chip]["json_node"], tab_fill)
    var chip_select_mask = 0xfff;
    var pos = current_lad_chips[current_chip_selected]["json_node"]["MIDAS_ID"];
    chip_select_mask &= ((~0x1) << pos);
    for (var i = 0; i < pos; ++i)
        chip_select_mask |= (0x1 << i);
    document.getElementById("debug").textContent = "0x" + chip_select_mask.toString(16);
    var roww = document.getElementById("pixel_parameters").insertRow(-1)
    var celll1 = roww.insertCell(0)
    var celll2 = roww.insertCell(1)
    celll1.textContent = "Predicted mask"
    celll2.textContent = "0x" + chip_select_mask.toString(16);

    reset_table("Bias_tab", 0)
    reset_table("Conf_tab", 0)
    reset_table("VDAC_tab", 0)
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
      tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
      tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    change_qc_image()
}

var ladder_clicked = function(ladname) {
    document.getElementById("ladder_header").style["visibility"] = "visible"
    lad = lads[ladname]["ladder"]
    document.getElementById("debug").textContent = lad.id + "-" + JSON.stringify(lads[ladname]["json_node"])
    change_state(lads[ladname], "selected", false)
    if (current_selected != "") {
        if (current_selected != ladname)
            change_state(lads[current_selected], "unselected", false)
    }
    current_selected = ladname
    reset_table("parameter_tab", 0)
    tab_fill = {"SpecBook ID : " : "SpecBookId", "FEB ID : " : "FEB_ID", "DAB channel : " : "FEB_channel"}
    fill_table_by_json("parameter_tab", lads[ladname]["json_node"], tab_fill)

    reset_table("ladder_lv_tab", 0)
    tab_fill = {"Hameg Module : " : "Module", "IP address : " : "IP", "Hameg Channel : " : "Channel"}
    fill_table_by_json("ladder_lv_tab", lads[ladname]["json_node"]["LV_parameters"], tab_fill)

    document.getElementById("ladder_lv_header").style["visibility"] = "visible"
    document.getElementById("ladder_lv_settings").style["visibility"] = "visible"

    document.getElementById("ladder_lv_status").setAttribute("data-odb-path", "/Equipment/HAMEG" + lads[ladname]["json_node"]["LV_parameters"]["Module"].toString()+ "/Variables/State[" + lads[ladname]["json_node"]["LV_parameters"]["Channel"].toString()  + "]")
    document.getElementById("ladder_lv_set_status").setAttribute("data-odb-path", "/Equipment/HAMEG" + lads[ladname]["json_node"]["LV_parameters"]["Module"].toString()+ "/Variables/Set State[" + lads[ladname]["json_node"]["LV_parameters"]["Channel"].toString()  + "]")
    document.getElementById("ladder_lv_read_volt").setAttribute("data-odb-path", "/Equipment/HAMEG" + lads[ladname]["json_node"]["LV_parameters"]["Module"].toString()+ "/Variables/Voltage[" + lads[ladname]["json_node"]["LV_parameters"]["Channel"].toString()  + "]")
    document.getElementById("ladder_lv_read_curr").setAttribute("data-odb-path", "/Equipment/HAMEG" + lads[ladname]["json_node"]["LV_parameters"]["Module"].toString()+ "/Variables/Current[" + lads[ladname]["json_node"]["LV_parameters"]["Channel"].toString()  + "]")


    /*mjsonrpc_db_get_values(["/Equipment/HAMEG" + lads[ladname]["json_node"]["LV_parameters"]["Module"].toString() + "/Variables/State[" + lads[ladname]["json_node"]["LV_parameters"]["Channel"].toString() + "]"]).then(function (rpc) {
        document.getElementById("debug").textContent = JSON.stringify(rpc.result)
        if (rpc.result.data[0] != null) {
            if (rpc.result.status[0] == true) {
                change_state(lads[ladname], "power_on", false)
            }
            else if (rpc.result.status[0] == false) {
                change_state(lads[ladname], "power_off", false)
            }
        }
    }).catch(function(error) {
       mjsonrpc_error_alert(error);
    });*/

    reset_table("ladder_hv_tab", 0)
    tab_fill = {"HV-box Module : " : "Module", "IP address : " : "IP", "HV-box Channel : " : "Channel"}
    fill_table_by_json("ladder_hv_tab", lads[ladname]["json_node"]["HV_parameters"], tab_fill)

    document.getElementById("ladder_hv_header").style["visibility"] = "visible"
    document.getElementById("ladder_hv_settings").innerHTML = ""

    document.getElementById("ladder_temperature").style["visibility"] = "visible"
    var port = lads[ladname]["json_node"]["Temperature_parameters"]["port"]
    var channela = lads[ladname]["json_node"]["Temperature_parameters"]["channel"]
    var input_channel = from_portchannel_to_input(port, channela)
    var link_temp = "/Equipment/Pixel_Temperatures/Variables/Input[" + input_channel + "]"
    document.getElementById("ladder_temp_read_volt").textContent = "Converted directly"
    //link_temp = "/Equipment/Pixel_/Converted/Temperature_" + input_channel
    document.getElementById("ladder_temp_read_c").setAttribute("data-odb-path", link_temp)

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
        chip.classList.add("chip_unselected")
        chip.textContent = "Chip " + ch.toString()
        let ch_num = ch.toString()
        chip.id = ch_num
        if (lads[ladname]["json_node"]["Chips"][ch_num]["Initial_Status"] == "Dead") {
            chip.classList.add("chip_dead")
            chip.textContent += "\nDEAD"
        }
        else if (lads[ladname]["json_node"]["Chips"][ch_num]["Initial_Status"] == "A") {
            //chip.classList.add("chip_available")
            chip.classList.add("chip_qc_a")
            chip.textContent += "\nA"
        }
        else if (lads[ladname]["json_node"]["Chips"][ch_num]["Initial_Status"] == "B") {
            //chip.classList.add("chip_available")
            chip.classList.add("chip_qc_b")
            chip.textContent += "\nB"
        }
        else if (lads[ladname]["json_node"]["Chips"][ch_num]["Initial_Status"] == "C") {
            //chip.classList.add("chip_available")
            chip.classList.add("chip_qc_c")
            chip.textContent += "\nC"
        }
        else if (lads[ladname]["json_node"]["Chips"][ch_num]["Initial_Status"] == "D") {
            //chip.classList.add("chip_available")
            chip.classList.add("chip_qc_d")
            chip.textContent += "\nD"
        }
        else if (lads[ladname]["json_node"]["Chips"][ch_num]["Initial_Status"] == "E") {
            //chip.classList.add("chip_available")
            chip.classList.add("chip_qc_e")
            chip.textContent += "\nE"
        }
        else if (lads[ladname]["json_node"]["Chips"][ch_num]["Initial_Status"] == "N.A.") {
            chip.classList.add("chip_available")
        }
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
    document.getElementById("qctestdiv").style["visibility"] = "hidden";
    reset_table("pixel_parameters", 0)
    reset_table("Bias_tab", 0)
    reset_table("Conf_tab", 0)
    reset_table("VDAC_tab", 0)
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
    document.getElementById(dacName + "_head").textContent = "Change values in " + "/Equipment/PixelsCentral/Settings/" + odb_dacName + "/" + parseInt(current_lad_chips[current_chip_selected]["json_node"]["MIDAS_ID"])
    reset_table(dacName + "_tab", 0)
    var table = document.getElementById(dacName + "_tab");
    table.style["visibility"] = "visible"
    dac_list = mupix_dacs[dacName]
    document.getElementById("debug").textContent = JSON.stringify(dac_list)
    var row0 = table.insertRow(0);
    row0.style.fontWeight="bold"
    var cell01 = row0.insertCell(0);
    cell01.innerHTML = "DAC name"
    var cell02 = row0.insertCell(1);
    cell02.innerHTML = "DAC value"
    var cell03 = row0.insertCell(2);
    cell03.innerHTML = "Default value"
    var cell04 = row0.insertCell(3);
    cell04.innerHTML = "Difference"
    var row_n = 1
    var watch_diff_cells = {}
    for (dac in dac_list) {
        var row1 = table.insertRow(row_n);
        var cell1 = row1.insertCell(0);
        var cell2 = row1.insertCell(1);
        cell1.innerHTML = dac;
        cell2.classList.add("modbvalue")
        var link = "/Equipment/PixelsCentral/Settings/" + odb_dacName + "/" + parseInt(current_lad_chips[current_chip_selected]["json_node"]["MIDAS_ID"]) + "/" + dac
        cell2.setAttribute("data-odb-path", link)
        cell2.setAttribute("data-odb-editable", "1")
        cell2.id = "DAC_tab_" + dac
        var cell3 = row1.insertCell(2);
        cell3.innerHTML = dac_list[dac];
        var cell4 = row1.insertCell(3);
        watch_diff_cells["DAC_tab_" + dac] = cell4
        cell2.addEventListener('DOMSubtreeModified', function() {
            dac_name = this.id.substr(8,this.id.length)
            watch_diff_cells[this.id].innerHTML = (this.textContent) - dac_list[dac_name];
            //watch_diff_cells[this.id].innerHTML = dac_name;
        })

        row_n++
    }
    evt.currentTarget.className += " active";
}

function configureChip(evt) {
    /*var mask_names = []
    var px_mask = []
    var num = (parseInt(lads[current_selected]["json_node"]["FEB_ID"])*12 + parseInt(current_lad_chips[current_chip_selected]["json_node"]["MIDAS_ID"]))
    for (var i = 0; i < 128; ++i) {
        mask_names.push("/Equipment/PixelsCentral/Settings/Daq/mask[" + i + "]")
        if (i === num)
            px_mask.push(false);
        else
            px_mask.push(true)
    }*/
    document.getElementById("pixel_configure_update").style["visibility"] = "visible"
    /*mjsonrpc_db_paste(mask_names, px_mask).then(function () {
        mjsonrpc_db_paste(["/Equipment/SwitchingCentral/Commands/MupixConfig"], [true])
    }).catch(function(error) {
        mjsonrpc_error_alert(error);
    });
    var chip_select_mask = 0xfff;
    var pos = current_lad_chips[current_chip_selected]["json_node"]["MIDAS_ID"];
    chip_select_mask &= ((~0x1) << pos);
    for (var i = 0; i < pos; ++i)
        chip_select_mask |= (0x1 << i);*/

    document.getElementById("pixel_configure_update_content").textContent = "Configuring chip number " + parseInt(current_lad_chips[current_chip_selected]["json_node"]["MIDAS_ID"])

    document.getElementById("debug").textContent = "0x" + chip_select_mask.toString(16);
    mjsonrpc_db_paste(["/Equipment/SwitchingCentral/Commands/MupixChipToConfigure"], [parseInt(current_lad_chips[current_chip_selected]["json_node"]["MIDAS_ID"])]).then(function () {
        mjsonrpc_db_paste(["/Equipment/SwitchingCentral/Commands/MupixConfig"], [true])
    }).catch(function(error) {
        mjsonrpc_error_alert(error);
    });
}

function resetDACs(evt) {
    var odb_dacName = ""
    var num = parseInt(current_lad_chips[current_chip_selected]["json_node"]["MIDAS_ID"])
    for (var dacName in mupix_dacs) {
        var daclist = []
        var dacvalues = []
        if (dacName === "Bias")
            odb_dacName = "BIASDACS"
        else if (dacName === "Conf")
            odb_dacName = "CONFDACS"
        else if (dacName === "VDAC")
            odb_dacName = "VDACS"
        for (var dac in mupix_dacs[dacName]) {
            daclist.push("/Equipment/PixelsCentral/Settings/" + odb_dacName + "/" + num + "/" + dac)
            dacvalues.push(mupix_dacs[dacName][dac])
        }
        document.getElementById("pixel_configure_update").style["visibility"] = "visible"
        document.getElementById("pixel_configure_update_content").textContent = "Resetting chip number " + num
        mjsonrpc_db_paste(daclist, dacvalues).catch(function(error) {
            mjsonrpc_error_alert(error);
         });
    }
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
                    num = parseInt(lads[lad_name]["json_node"]["Chips"][chip]["MIDAS_ID"])
                    mask_names.push("/Equipment/PixelsCentral/Settings/Daq/mask[" + num + "]")
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
        mjsonrpc_db_paste(["/Equipment/SwitchingCentral/Commands/MupixConfig"], [true])
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

var create_ladder_legend = function () {
    document.getElementById("debug").textContent = "hmmm"
    for (state in states) {
        let lad = document.createElement("div")
        lad.type = "submit"
        lad.style = "width:70px;height:20px;position: relative;"
        lad.classList.add("ladderDiv")
        lad.classList.add("ladder_unselected")
        lad.id = "ladder_legend_" + state
        document.getElementById("debug").textContent += "-" + states[state]
        var state_name = states[state].substr(7, states[state].length)
        lad.textContent = state_name
        ladder = create_ladder_element(lad)
        change_state(ladder, state_name, false)
        //change_state(ladder, "hv_on", true)
        document.getElementById("ladder_legend").appendChild(lad)
        document.getElementById("debug").textContent += "-" + state_name
        var linebreak = document.createElement("div");
        linebreak.classList.add("line_break_white")
        document.getElementById("ladder_legend").appendChild(linebreak)
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
            var linebreak = document.createElement("div");
            linebreak.classList.add("line_break")
            ele.appendChild(linebreak)
        }
    }
}

var change_qc_image = function () {

    /*var xmlhttp = new XMLHttpRequest();
    //xmlhttp.responseType = "arraybuffer";
    xmlhttp.onreadystatechange = function(){
      if(xmlhttp.status==200 && xmlhttp.readyState==4){
        document.getElementById("QC").src = xmlhttp.responseText;
        //document.getElementById("QC").setAttribute('src', "http://mu3ebe:8081/home/mu3e/online/online/pixels/qctest/ladder/output/IV_"+String((current_chip_selected))+".png")
    }
      else {
          document.getElementById("QC").textContent = "NOOOOOOO IMAGE!!!" ;
      }
    }
    xmlhttp.open("GET","http://mu3ebe:8081//home/mu3e/online/online/pixels/qctest/ladder/output/IV_"+String((current_chip_selected))+".png",true);
    xmlhttp.send();*/
    lad_id = parseInt(parseInt(current_lad_chips[current_chip_selected]["json_node"]["MIDAS_ID"])/3)
    if (lad_id > 17)
        lad_id += 2
    document.getElementById("QC").src="http://mu3ebe:8081//home/mu3e/online/online/pixels/qctest/ladder/ana_output/ladder_" + lad_id  + "_chip_" + parseInt(current_lad_chips[current_chip_selected]["json_node"]["MIDAS_ID"])%3 + ".png";
    document.getElementById("qctestdiv").style["visibility"] = "visible";

}

var update_qc_dacs_link = function() {
    chip_id = current_lad_chips[current_chip_selected]["json_node"]["MIDAS_ID"]
    opt_vpvco = current_lad_chips[current_chip_selected]["json_node"]["opt_VPVCO"]
    opt_vnvco = current_lad_chips[current_chip_selected]["json_node"]["opt_VNVCO"]
    if (opt_vpvco == 0) {
        document.getElementById("pixel_configure_update_content").textContent = "VPVCO and VNVCO not optimized for chip " + chip_id
        return
    }
    else {
        link_vpvco = "/Equipment/PixelsCentral/Settings/BIASDACS/" + parseInt(current_lad_chips[current_chip_selected]["json_node"]["MIDAS_ID"]) + "/VPVCO"
        link_vnvco = "/Equipment/PixelsCentral/Settings/BIASDACS/" + parseInt(current_lad_chips[current_chip_selected]["json_node"]["MIDAS_ID"]) + "/VNVCO"
        mjsonrpc_db_paste([link_vpvco, link_vnvco], [opt_vpvco, opt_vnvco]).then(function () {
            document.getElementById("pixel_configure_update_content").textContent = "VPVCO for chip " + chip_id + " set to " + opt_vpvco
            document.getElementById("pixel_configure_update_content").textContent += ", VNVCO for chip " + chip_id + " set to " + opt_vnvco
        }).catch(function(error) {
            mjsonrpc_error_alert(error);
        });
        }
}

var setup_listeners = function () {
    document.getElementById("ladder_temp_read_volt").addEventListener('DOMSubtreeModified', function () {
        document.getElementById("ladder_temp_read_c").innerHTML = convert_v_to_temperature(this.textContent);
    })
}

var setup = function () {
    create_octagon(document.getElementById("L0_US"), "L0_US_L", false)
    create_decagon(document.getElementById("L1_US"), "L1_US_L", false)
    create_octagon(document.getElementById("L0_DS"), "L0_DS_L", true)
    create_decagon(document.getElementById("L1_DS"), "L1_DS_L", true)

    list_chips_configuration(document.getElementById("multiple_chip_configuration_us"), "Upstream")
    list_chips_configuration(document.getElementById("multiple_chip_configuration_ds"), "Downstream")

    //setup_listeners()
    create_ladder_legend()
}

var load_json = function () {
    var xmlhttp2 = new XMLHttpRequest();
    xmlhttp2.onreadystatechange = function(){
      if(xmlhttp2.status==200 && xmlhttp2.readyState==4){
        var words = xmlhttp2.responseText;//.split(' ');
        document.getElementById("debug").textContent += "OPEN! - ";
        document.getElementById("debug").textContent += words;
        mupix_dacs = JSON.parse(words);
      }
      else {
          document.getElementById("debug").textContent += "NOOOOOOO!!!" ;
      }
    }

    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function(){
      if(xmlhttp.status==200 && xmlhttp.readyState==4){
        var words = xmlhttp.responseText;//.split(' ');
        json_config = JSON.parse(words);
        xmlhttp2.open("GET","mupix_default_dacs.json",true);
        xmlhttp2.send();
        setup();
      }
      else {
          document.getElementById("debug").textContent += "NOOOOOOO!!!" ;
      }
    }
    xmlhttp.open("GET","mupix_configuration.json",true);
    xmlhttp.send();
}

load_json()

