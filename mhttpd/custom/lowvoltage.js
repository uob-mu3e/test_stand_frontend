var PathH1='"/Equipment/HAMEG1/Variables/Set State'
var PathH2='"/Equipment/HAMEG2/Variables/Set State'
var PathH3='"/Equipment/HAMEG3/Variables/Set State'
var PathH4='"/Equipment/HAMEG4/Variables/Set State'
var PathH5='"/Equipment/HAMEG5/Variables/Set State'
var PathH6='"/Equipment/HAMEG6/Variables/Set State'
var PathH7='"/Equipment/HAMEG7/Variables/Set State'
var PathH8='"/Equipment/HAMEG8/Variables/Set State'
var PathH9='"/Equipment/HAMEG9/Variables/Set State'
var PathTDK='"/Equipment/Genesys/Variables/Set State'
var PathSciFi='"/Equipment/SciFi LV/Variables/ChState'

function SciFiUSOFF(){
var clist=document.getElementsByName("CheckboxSciFi");
// Stupid but works
console.log("SciFi US Off!");
modbset("/Equipment/SciFi LV/Variables/ChState[0]", 0);
modbset("/Equipment/SciFi LV/Variables/ChState[1]", 0);
modbset("/Equipment/SciFi LV/Variables/ChState[2]", 0);
}

function SciFiUSON(){
var clist=document.getElementsByName("CheckboxSciFi");
// Stupid but works
console.log("SciFi US Off!");
modbset("/Equipment/SciFi LV/Variables/ChState[0]", 1);
modbset("/Equipment/SciFi LV/Variables/ChState[1]", 1);
modbset("/Equipment/SciFi LV/Variables/ChState[2]", 1);
}

function SciFiDSOFF(){
var clist=document.getElementsByName("CheckboxSciFi");
// Stupid but works
console.log("SciFi DS Off!");
modbset("/Equipment/SciFi LV/Variables/ChState[3]", 0);
modbset("/Equipment/SciFi LV/Variables/ChState[4]", 0);
modbset("/Equipment/SciFi LV/Variables/ChState[5]", 0);
}

function SciFiDSON(){
var clist=document.getElementsByName("CheckboxSciFi");
// Stupid but works
console.log("SciFi DS On!");
modbset("/Equipment/SciFi LV/Variables/ChState[3]", 1);
modbset("/Equipment/SciFi LV/Variables/ChState[4]", 1);
modbset("/Equipment/SciFi LV/Variables/ChState[5]", 1);
}

function SciFiOFF(){
var clist=document.getElementsByName("CheckboxSciFi");
// Stupid but works
console.log("SciFi Off!");
modbset("/Equipment/SciFi LV/Variables/ChState[0]", 0);
modbset("/Equipment/SciFi LV/Variables/ChState[1]", 0);
modbset("/Equipment/SciFi LV/Variables/ChState[2]", 0);
modbset("/Equipment/SciFi LV/Variables/ChState[3]", 0);
modbset("/Equipment/SciFi LV/Variables/ChState[4]", 0);
modbset("/Equipment/SciFi LV/Variables/ChState[5]", 0);
}

function SciFiON(){
var clist=document.getElementsByName("CheckboxSciFi");
// Stupid but works
console.log("SciFi On!");
modbset("/Equipment/SciFi LV/Variables/ChState[0]", 1);
modbset("/Equipment/SciFi LV/Variables/ChState[1]", 1);
modbset("/Equipment/SciFi LV/Variables/ChState[2]", 1);
modbset("/Equipment/SciFi LV/Variables/ChState[3]", 1);
modbset("/Equipment/SciFi LV/Variables/ChState[4]", 1);
modbset("/Equipment/SciFi LV/Variables/ChState[5]", 1);
}

function PixelsOFF() {
var clist=document.getElementsByName("CheckboxPixels");
for (var i = 0; i < clist.length; ++i) { clist[i].checked = false; }  

mjsonrpc_db_paste([PathH1 +'[0]"', PathH1 +'[1]"', PathH1 +'[2]"', PathH1 +'[3]"', PathH2 +'[0]"', PathH2 +'[1]"', PathH2 +'[2]"', PathH2 +'[3]"',
PathH3 +'[0]"', PathH3 +'[1]"', PathH3 +'[2]"', PathH3 +'[3]"', PathH4 +'[0]"', PathH4 +'[1]"', PathH4 +'[2]"', PathH4 +'[3]"',
PathH5 +'[0]"', PathH5 +'[1]"', PathH5 +'[2]"', PathH5 +'[3]"', PathH6 +'[0]"', PathH6 +'[1]"', PathH6 +'[2]"', PathH6 +'[3]"',
PathH7 +'[0]"', PathH7 +'[1]"', PathH7 +'[2]"', PathH7 +'[3]"', PathH8 +'[0]"', PathH8 +'[1]"', PathH8 +'[2]"', PathH8 +'[3]"',
PathH9 +'[0]"', PathH9 +'[1]"', PathH9 +'[2]"', PathH9 +'[3]"'], [0]).then(function(rpc) {

        }).catch(function(error) {
            mjsonrpc_error_alert(error);
        });

}

//How many channels are used???
function FebOFF() {
var clist=document.getElementsByName("CheckboxFEB");
for (var i = 0; i < clist.length; ++i) { clist[i].checked = false; }  

mjsonrpc_db_paste([PathTDK + '[0]"'], [0]).then(function(rpc) {

        }).catch(function(error) {
            mjsonrpc_error_alert(error);
        });

}

function AllOFF() {
var clist=document.getElementsByClassName("modbcheckbox");
for (var i = 0; i < clist.length; ++i) { clist[i].checked = false; }  

mjsonrpc_db_paste([PathSciFi + '[0]"', PathSciFi + '[1]"', PathSciFi + '[2]"', PathSciFi + '[3]"', PathSciFi + '[4]"', PathSciFi + '[5]"',
PathH1 +'[0]"', PathH1 +'[1]"', PathH1 +'[2]"', PathH1 +'[3]"', PathH2 +'[0]"', PathH2 +'[1]"', PathH2 +'[2]"', PathH2 +'[3]"',
PathH3 +'[0]"', PathH3 +'[1]"', PathH3 +'[2]"', PathH3 +'[3]"', PathH4 +'[0]"', PathH4 +'[1]"', PathH4 +'[2]"', PathH4 +'[3]"',
PathH5 +'[0]"', PathH5 +'[1]"', PathH5 +'[2]"', PathH5 +'[3]"', PathH6 +'[0]"', PathH6 +'[1]"', PathH6 +'[2]"', PathH6 +'[3]"',
PathH7 +'[0]"', PathH7 +'[1]"', PathH7 +'[2]"', PathH7 +'[3]"', PathH8 +'[0]"', PathH8 +'[1]"', PathH8 +'[2]"', PathH8 +'[3]"',
PathH9 +'[0]"', PathH9 +'[1]"', PathH9 +'[2]"', PathH9 +'[3]"', PathTDK + '[0]"', PathTDK + '[1]"'], [0]).then(function(rpc) {

        }).catch(function(error) {
            mjsonrpc_error_alert(error);
        });

}

var json_configuration = {}

var get_description = function(hameg, channel) {
    var descr = "Finding"
    var found = false
    var node_js = json_configuration["Stations"]["Tracker"]["Direction"]
    for (direction in node_js) {
        for (layer in node_js[direction]["Layers"]) {
            for (ladder in node_js[direction]["Layers"][layer]["Ladders"]) {
                if (node_js[direction]["Layers"][layer]["Ladders"][ladder]["LV_parameters"]["Module"] === hameg && node_js[direction]["Layers"][layer]["Ladders"][ladder]["LV_parameters"]["Channel"] === channel) {
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

var change_descriptions = function () {
    for (var hameg = 0; hameg < 9; ++hameg) {
        var tab = document.getElementById("hameg" + hameg.toString())
        for (var channel = 0; channel < 4; ++channel) {
            //tab.rows[2+channel].cells[7].innerHTML = get_description(hameg,channel)
            continue;
        }
    }
}

var load_json = function () {
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function(){
      if(xmlhttp.status==200 && xmlhttp.readyState==4){
        var words = xmlhttp.responseText;//.split(' ');
        json_configuration = JSON.parse(words);
        change_descriptions()
      }
    }
    xmlhttp.open("GET","mupix_configuration.json",true);
    xmlhttp.send();
}
load_json()
