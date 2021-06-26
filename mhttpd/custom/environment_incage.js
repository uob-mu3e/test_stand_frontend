var scheme = {
    0: { "Env_US1 - Water": "undefined" },
    1: { "Env_US1 - PCMini52_RH": "undefined" },
    2: { "Env_US1 - PCMini52_T": "undefined" },
    3: { "Env_US2 - O2-7-top": "undefined" },
    4: { "Env_US2 - HIH4040_top": "undefined" },
    5: { "Env_US2 - LM35_Fibre": "undefined" },
    6: { "Env_US2 - LM35_FEC1": "undefined" },
    7: { "Env_US2 - LM35_top": "undefined" },

    9: { "Env_DS1 - Water": "undefined" },
    11: { "Env_DS2 - O2-4-top": "undefined" },
    12: { "Env_DS2 - HIH4040_top": "undefined" },
    13: { "Env_DS2 - LM35_Fibre": "undefined" },
    14: { "Env_DS2 - LM35_FEC1": "undefined" },
    15: { "Env_DS2 - LM35_top": "undefined" },

    18: { "Env_DS3 - O2-11-bottom": "undefined" },
    19: { "Env_US3 - O2-6-central": "undefined" },
    20: { "Env_US3 - HIH4040_bottom": "undefined" },
    21: { "Env_US3 - HIH4040_central": "undefined" },
    23: { "Env_US6 - O2_3": "undefined" },
    24: { "Env_US6 - LM35_3": "undefined" },
    25: { "Env_US6 - HIH_3": "undefined" },

    27: { "Env_DS3 - O2-11-bottom": "undefined" },
    28: { "Env_DS3 - O2-6-central": "undefined" },
    29: { "Env_DS3 - HIH4040_bottom": "undefined" },
    30: { "Env_DS3 - HIH4040_central": "undefined" },
    32: { "Env_DS6 - LM35_6": "undefined" },
    33: { "Env_DS6 - HIH_6": "undefined" },
    34: { "Env_DS6 - O2_6": "undefined" }
}

var table = document.getElementById("env_tab")

for (var key in scheme) {
    var row1 = table.insertRow(-1);
    var cell1 = row1.insertCell(0);
    cell1.textContent = key
    var cell2 = row1.insertCell(1);
    var cell3 = row1.insertCell(2);
    var cell4 = row1.insertCell(3);
    for (var keya in scheme[key]) {
        cell2.textContent = keya

        cell3.classList.add("modbvalue")
        var link = "/Equipment/Environment/Variables/Input[" + key + "]"
        cell3.setAttribute("data-odb-path", link)

        cell4.classList.add("modbvalue")
        var link2 = "/Equipment/Environment/Converted/" + keya
        cell4.setAttribute("data-odb-path", link2)
    }
}