function generateTableHead(table, febname){
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

function generateTable(table){

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
    }

    let row7 = table.insertRow();
    let cell71 = row7.insertCell();
    //cell71.setAttribute("colspan","3");
    let text71 = document.createTextNode("Delay");
    cell71.appendChild(text71);

    let cell72 = row7.insertCell();
    //cell72.setAttribute("colspan","2");
    cell72.setAttribute("name","modbvalue");
    cell72.setAttribute("data-odb-path","/Equipment/Switching/Settings/Sorter Delay[0]");
    cell72.setAttribute("data-odb-editable","1");

    let cell73 = row7.insertCell();
    cell73.setAttribute("colspan","2");
    let text73 = document.createTextNode("Hits out");
    cell73.appendChild(text73);

    let cell74 = row7.insertCell();
    cell74.setAttribute("colspan","2");
    let text74 = document.createTextNode("123456");
    cell74.appendChild(text74);

    let cell75 = row7.insertCell();
    cell75.setAttribute("colspan","2");
    let text75 = document.createTextNode("Hits inside");
    cell75.appendChild(text75);

    let cell76 = row7.insertCell();
    cell76.setAttribute("colspan","2");
    let text76 = document.createTextNode("723456");
    cell76.appendChild(text76);

    let cell77 = row7.insertCell();
    //cell77.setAttribute("colspan","2");
    let text77 = document.createTextNode("Credits");
    cell77.appendChild(text77);

    let cell78 = row7.insertCell();
    let text78 = document.createTextNode("123");
    cell78.appendChild(text78);


}

let table = document.getElementById("febtable");
generateTable(table);
generateTableHead(table, "FEB 1");




function update_slots(valuex){
    var value = valuex;
    if(typeof valuex === 'string')
        value = JSON.parse(valuex);


}

