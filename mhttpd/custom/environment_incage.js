var scheme = {
    0: {"Env_US1 - Water": "Water"} ,
    1: {"Env_US1 - PCMini52_RH": "PCMini52RH"} ,
    2: {"Env_US1 - PCMini52_T": "PCMini52T" } ,
    3: {"Env_US2 - O2-7-top": "O2" } ,
    4: {"Env_US2 - HIH4040_top": "Honeywell" } ,
    5: {"Env_US2 - LM35_Fibre": "LM35" } ,
    6: {"Env_US2 - LM35_FEC1": "LM35" } ,
    7: {"Env_US2 - LM35_top": "LM35" } ,

    9: { "Env_DS1 - Water": "Water" } ,
    11: { "Env_DS2 - O2-4-top": "O2" } ,
    12: {"Env_DS2 - HIH4040_top": "Honeywell" } ,
    13: {"Env_DS2 - LM35_Fibre": "LM35" } ,
    14: {"Env_DS2 - LM35_FEC1": "LM35" } ,
    15: {"Env_DS2 - LM35_top": "LM35" } ,

    18: { "Env_DS3 - O2-11-bottom": "O2" } ,
    19: { "Env_US3 - O2-6-central": "O2" } ,
    20: {"Env_US3 - HIH4040_bottom": "Honeywell" } ,
    21: {"Env_US3 - HIH4040_central": "Honeywell" } ,
    //{23: {"Env_US6 - O2_3": "undefined" },
    //{24: {"Env_US6 - LM35_3": "undefined" },
    //{25: {"Env_US6 - HIH_3": "undefined" },

    27: { "Env_DS3 - O2-11-bottom": "O2" } ,
    28: { "Env_DS3 - O2-6-central": "O2" } ,
    29: {"Env_DS3 - HIH4040_bottom": "Honeywell" } ,
    30: {"Env_DS3 - HIH4040_central": "Honeywell" }
    //{32: {"Env_DS6 - LM35_6": "undefined" },
    //{33: {"Env_DS6 - HIH_6": "undefined" },
    //{34: {"Env_DS6 - O2_6": "undefined" }}
}

var table = document.getElementById("env_tab")

for (var key in scheme) {
    var row1 = table.insertRow(-1);
    var cell1 = row1.insertCell(0);
    cell1.textContent = key
    var cell2 = row1.insertCell(1);
    var cell3 = row1.insertCell(2);
    var cell4 = row1.insertCell(3);
    var cell5 = row1.insertCell(4);
    for (var keya in scheme[key]) {
        cell2.textContent = keya

        cell3.classList.add("modbvalue")
        var link = "/Equipment/Environment/Variables/Input[" + key + "]"
        cell3.setAttribute("data-odb-path", link)

        cell4.classList.add("modbvalue")
        var link2 = "/Equipment/Environment/Converted/" + keya
        cell4.setAttribute("data-odb-path", link2)
        if (scheme[key][keya] == "Honeywell" || scheme[key][keya] == "PCMini52RH")
            cell5.textContent = "%RH"
        else if (scheme[key][keya] == "LM35" || scheme[key][keya] == "PCMini52T")
            cell5.textContent = "C"
        else if (scheme[key][keya] == "O2")
            cell5.textContent = "%"
        else if (scheme[key][keya] == "Water")
            cell5.textContent = "(bool)"
    }
}

var row2 = table.insertRow(-1);
var cell21 = row2.insertCell(0);
cell21.textContent = "-"
var cell22 = row2.insertCell(1);
var cell23 = row2.insertCell(2);
var cell24 = row2.insertCell(3);
var cell25 = row2.insertCell(4);
cell22.textContent = "Dew Point"
cell23.textContent = "-"
cell24.classList.add("modbvalue")
var link22 = "/Equipment/Environment/Converted/Computed/DewPoint"
cell24.setAttribute("data-odb-path", link22)
cell25.textContent = "C"