function generateLVDSMupixTableHead(table, febname){
    let thead = table.createTHead();
    let row1 = thead.insertRow();
    let th = document.createElement("th");
    th.setAttribute("colspan","12");
    let text = document.createTextNode(febname);
    th.appendChild(text);
    row1.appendChild(th);

    let row2 = thead.insertRow();

    th = document.createElement("th");
    text = document.createTextNode("LVDS ID");
    th.appendChild(text);
    row2.appendChild(th);

    th = document.createElement("th");
    text = document.createTextNode("RX READY");
    th.appendChild(text);
    row2.appendChild(th);

    th = document.createElement("th");
    text = document.createTextNode("RX STATE");
    th.appendChild(text);
    row2.appendChild(th);

    th = document.createElement("th");
    text = document.createTextNode("PLL LOCK");
    th.appendChild(text);
    row2.appendChild(th);

    th = document.createElement("th");
    text = document.createTextNode("DISP ERROR");
    th.appendChild(text);
    row2.appendChild(th);

    th = document.createElement("th");
    text = document.createTextNode("Num Hits");
    th.appendChild(text);
    row2.appendChild(th);

    th = document.createElement("th");
    text = document.createTextNode("Num Mupix Hits");
    th.appendChild(text);
    row2.appendChild(th);
}


function generateLVDSMupixTable(table, counters, feb){

    for(var lvds=0; lvds < 36; lvds++){
        let row = table.insertRow();
        for(var i=0; i < 7; i++){
            let cell = row.insertCell();
            let text = document.createTextNode(0);
            cell.appendChild(text);
            counters.push(text);
        }
    }
}


function generateSwitchingTableHead(table){
    let thead = table.createTHead();
    let row1 = thead.insertRow();
    let th = document.createElement("th");
    th.setAttribute("colspan","12");
    let text = document.createTextNode("Switching Board");
    th.appendChild(text);
    row1.appendChild(th);
}


function generateSwitchingTable(table, counters){

    let row1 = table.insertRow();

    let cell11 = row1.insertCell();
    cell11.setAttribute("colspan","2");
    let text11 = document.createTextNode("Stream FIFO full");
    cell11.appendChild(text11);

    let cell12 = row1.insertCell();
    let text12 = document.createTextNode(0);
    cell12.appendChild(text12);
    counters.push(text12);

    let cell13 = row1.insertCell();
    cell13.setAttribute("colspan","2");
    let text13 = document.createTextNode("Bank builder idle not header");
    cell13.appendChild(text13);

    let cell14 = row1.insertCell();
    let text14 = document.createTextNode(0);
    cell14.appendChild(text14);
    counters.push(text14);

    let cell15 = row1.insertCell();
    let text15 = document.createTextNode("Bank builder RAM full");
    cell15.appendChild(text15);

    let cell16 = row1.insertCell();
    let text16 = document.createTextNode(0);
    cell16.appendChild(text16);
    counters.push(text16);

    let cell17 = row1.insertCell();
    let text17 = document.createTextNode("Bank builder tag FIFO full");
    cell17.appendChild(text17);

    let cell18 = row1.insertCell();
    let text18 = document.createTextNode(0);
    cell18.appendChild(text18);
    counters.push(text18);

    let row2 = table.insertRow();

    let cell21 = row2.insertCell();
    let text21 = document.createTextNode("FEB");
    cell21.appendChild(text21);


    let cell22 = row2.insertCell();
    let text22 = document.createTextNode("FIFO");
    cell22.appendChild(text22);
    cell22.appendChild(document.createElement("br"));
    let text22b = document.createTextNode("almost full");
    cell22.appendChild(text22b);

    let cell23 = row2.insertCell();
    let text23 = document.createTextNode("FIFO");
    cell23.appendChild(text23);
    cell23.appendChild(document.createElement("br"));
    let text23b = document.createTextNode("full");
    cell23.appendChild(text23b);

    cell22.appendChild(text22b);
    let cell24 = row2.insertCell();
    let text24 = document.createTextNode("Skip events");
    cell24.appendChild(text24);

    let cell25 = row2.insertCell();
    let text25 = document.createTextNode("Events");
    cell25.appendChild(text25);

    let cell26 = row2.insertCell();
    let text26 = document.createTextNode("Subheaders");
    cell26.appendChild(text26);

    let cell27 = row2.insertCell();
    let text27 = document.createTextNode("Merger rate");
    cell27.appendChild(text27);

    let cell28 = row2.insertCell();
    let text28 = document.createTextNode("Reset phase");
    cell28.appendChild(text28);

    let cell29 = row2.insertCell();
    let text29 = document.createTextNode("TX Reset");
    cell29.appendChild(text29);

    for(var feb=0; feb < 12; feb++){
        let row = table.insertRow();
        for(var i=0; i < 10; i++){
            let cell = row.insertCell();
            let text = document.createTextNode(0);
            cell.appendChild(text);
            counters.push(text);
        }
    }
}


function generateSorterTableHead(table, febname){
    let thead = table.createTHead();
    let row1 = thead.insertRow();
    let th = document.createElement("th");
    th.setAttribute("colspan","12");
    let text = document.createTextNode(febname);
    th.appendChild(text);
    row1.appendChild(th);

    let row2 = thead.insertRow();
    for(var i=1; i <= 12; i++){
        let th = document.createElement("th");
        let text = document.createTextNode("Input" + i);
        th.appendChild(text);
        row2.appendChild(th);
    }
}


function generateSorterTable(table, counters, fractions, inside){

    let row1 = table.insertRow();
    let cell1 = row1.insertCell();
    cell1.setAttribute("colspan","12");
    cell1.setAttribute("style","text-align:center");
    let text = document.createTextNode("In time");
    cell1.appendChild(text);

    let row2 = table.insertRow();
    for(var i=1; i <= 12; i++){
        let cell = row2.insertCell();
        let text = document.createTextNode(345678+17*i);
        cell.appendChild(text);
        counters.push(text);
    }

    let row3 = table.insertRow();
    let cell3 = row3.insertCell();
    cell3.setAttribute("colspan","12");
    cell3.setAttribute("style","text-align:center");
    let text3 = document.createTextNode("Out of time");
    cell3.appendChild(text3);

    let row4 = table.insertRow();
    for(i=1; i <= 12; i++){
        let cell = row4.insertCell();
        let text = document.createTextNode(1234+665*i);
        cell.appendChild(text);
        counters.push(text);
    }

    let row5 = table.insertRow();
    let cell5 = row5.insertCell();
    cell5.setAttribute("colspan","12");
    cell5.setAttribute("style","text-align:center");
    let text5 = document.createTextNode("In-time fraction");
    cell5.appendChild(text5);

    let row6 = table.insertRow();
    for(i=1; i <= 12; i++){
        let cell = row6.insertCell();
        let text = document.createTextNode(0.999);
        cell.appendChild(text);
        fractions.push(text);
    }

    let row7 = table.insertRow();
    let cell7 = row7.insertCell();
    cell7.setAttribute("colspan","12");
    cell7.setAttribute("style","text-align:center");
    let text7 = document.createTextNode("Overflows");
    cell7.appendChild(text7);

    let row8 = table.insertRow();
    for(i=1; i <= 12; i++){
        let cell = row8.insertCell();
        let text = document.createTextNode(456);
        cell.appendChild(text);
        counters.push(text);
    }

    let row9 = table.insertRow();
    let cell91 = row9.insertCell();
    let text91 = document.createTextNode("Delay");
    cell91.appendChild(text91);

    let cell92 = row9.insertCell();
    cell92.setAttribute("name","modbvalue");
    cell92.setAttribute("data-odb-path","/Equipment/Switching/Settings/Sorter Delay[0]");
    cell92.setAttribute("data-odb-editable","1");

    let cell93 = row9.insertCell();
    cell93.setAttribute("colspan","2");
    let text93 = document.createTextNode("Hits out");
    cell93.appendChild(text93);

    let cell94 = row9.insertCell();
    cell94.setAttribute("colspan","2");
    let text94 = document.createTextNode("123456");
    cell94.appendChild(text94);
    counters.push(text94);

    let cell95 = row9.insertCell();
    cell95.setAttribute("colspan","2");
    let text95 = document.createTextNode("Hits inside");
    cell95.appendChild(text95);

    let cell96 = row9.insertCell();
    cell96.setAttribute("colspan","2");
    let text96 = document.createTextNode("723456");
    cell96.appendChild(text96);
    inside.push(text96);

    let cell97 = row9.insertCell();
    //cell97.setAttribute("colspan","2");
    let text97 = document.createTextNode("Credits");
    cell97.appendChild(text97);

    let cell98 = row9.insertCell();
    let text98 = document.createTextNode("123");
    cell98.appendChild(text98);
    counters.push(text98);

}

var counters = [];
var fractions = [];
var inside = [];

var switchingcounters = [];

var mupixlvdsstatus = [];


function init(){
    let main =  document.getElementById("mmain");

    // Switching counter table
    let swdiv =  document.createElement("div");
    swdiv.setAttribute("style","float:left, margin-right=20%");
    main.appendChild(swdiv);

    let swtab = document.createElement("table");
    swtab.setAttribute("class","mtable");
    swtab.setAttribute("style","margin-left: 0.5em");
    generateSwitchingTable(swtab, switchingcounters);
    generateSwitchingTableHead(swtab);

    swdiv.appendChild(swtab);


    // Loop over pixel FEBs
    for(var i=0; i < 10; i++){
        let div =  document.createElement("div");
        if ( i === 9){
            div.setAttribute("style","float:left, margin-right=20%");
        } else {
            div.setAttribute("style","float:left");
        }
        main.appendChild(div);

        counters.push([]);
        fractions.push([]);
        inside.push([]);

        let tab = document.createElement("table");
        tab.setAttribute("class","mtable");
        tab.setAttribute("style","margin-left: 0.5em");
        generateSorterTable(tab, counters[i], fractions[i], inside[i]);
        generateSorterTableHead(tab, "FEB "+i);

        div.appendChild(tab);
    }

    // Pixel FEBs LVDS Links
    for(var feb=0; feb < 10; feb++){
        let lvdsdiv =  document.createElement("div");
        lvdsdiv.setAttribute("style","float:left");
        main.appendChild(lvdsdiv);

        let lvdstab = document.createElement("table");
        lvdstab.setAttribute("class","mtable");
        lvdstab.setAttribute("style","margin-left: 0.5em");
        generateLVDSMupixTable(lvdstab, mupixlvdsstatus, feb);
        generateLVDSMupixTableHead(lvdstab, "LVDS Mupix FEB "+feb)

        lvdsdiv.appendChild(lvdstab);
    }


    mjsonrpc_db_get_values(["/Equipment/Switching/Variables/SCSO"]).then(function(rpc) {
        update_scso(rpc.result.data[0]);
     }).catch(function(error) {
        mjsonrpc_error_alert(error);
     });

    mjsonrpc_db_get_values(["/Equipment/Switching/Variables/SCCN"]).then(function(rpc) {
        update_sccn(rpc.result.data[0]);
     }).catch(function(error) {
        mjsonrpc_error_alert(error);
     });

    mjsonrpc_db_get_values(["/Equipment/Mupix/Variables/PSLL"]).then(function(rpc) {
        update_psll(rpc.result.data[0]);
     }).catch(function(error) {
        mjsonrpc_error_alert(error);
     });

}

init();

function update_scso(valuex){
    var value = valuex;
    if(typeof valuex === 'string')
        value = JSON.parse(valuex);

    for(var feb=0; feb < 10; feb++){
        var sum = 0;
        for(var input=0; input <12; input++){
            counters[feb][input].nodeValue=value[feb*39 + input + 1];
            sum += value[feb*39 + input + 1];
            counters[feb][12+input].nodeValue = value[feb*39 +12 + input + 1];
            counters[feb][24+input].nodeValue = value[feb*39 +24 + input + 1];
            sum -= value[feb*39 +24 + input + 1];
            counters[feb][36].nodeValue = value[feb*39 + 37];
            sum -= value[feb*39 + 37];
            counters[feb][37].nodeValue = value[feb*39 + 38];

            var frac = value[feb*39 + input + 1] / (value[feb*39 + input + 1] + value[feb*39 +12 + input + 1]);
            fractions[feb][input].nodeValue = frac.toFixed(4);
            if(frac > 0.99)
               fractions[feb][input].parentNode.style.color = "green";
            else
               fractions[feb][input].parentNode.style.color = "red";
        }
        inside[feb][0].nodeValue = sum;
    }
}

function update_sccn(valuex){
    var value = valuex;
    if(typeof valuex === 'string')
        value = JSON.parse(valuex);

    console.log(value);

    for(var input=0; input <112; input++){
        switchingcounters[input].nodeValue = value[input];
    }
}

function update_psll(valuex){
    var value = valuex;
    if(typeof valuex === 'string')
        value = JSON.parse(valuex);

    console.log(value);

    for(var feb=0; feb < 10; feb++){
        for(var lvds=0; lvds < 36; lvds++){
            for(var i=0; i < 7; i++){
                if ( i >= 1 && i <= 4 ) {
                    if ( i === 1 ) {
                        // RX READY
                        mupixlvdsstatus[i+feb*36*7+lvds*7].nodeValue = ((value[1+feb*36*4+lvds*4] >> 31) & 0x1).toString(16);
                    }
                    if ( i === 2 ) {
                        // RX STATE
                        mupixlvdsstatus[i+feb*36*7+lvds*7].nodeValue = ((value[1+feb*36*4+lvds*4] >> 29) & 0x3).toString(16);
                    }
                    if ( i === 3 ) {
                        // PLL LOCK
                        mupixlvdsstatus[i+feb*36*7+lvds*7].nodeValue = ((value[1+feb*36*4+lvds*4] >> 28) & 0x1).toString(16);
                    }
                    if ( i === 4 ) {
                        // DISP ERROR
                        mupixlvdsstatus[i+feb*36*7+lvds*7].nodeValue = (value[1+feb*36*4+lvds*4] & 0x0FFFFFFF).toString(16);
                    }
                } else if ( i > 4){
                    mupixlvdsstatus[i+feb*36*7+lvds*7].nodeValue = value[i-3+feb*36*4+lvds*4];
                } else {
                    mupixlvdsstatus[i+feb*36*7+lvds*7].nodeValue = (value[i+feb*36*4+lvds*4]);
                }
            }
        }
    }

}
