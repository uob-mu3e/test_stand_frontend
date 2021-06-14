
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

function generateTable(table, counters, fractions, inside){

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

var counters =[];
var fractions =[];
var inside=[];

function init(){
   let main =  document.getElementById("mmain");
    // Loop over pixel FEBs
    for(var i=0; i < 10; i++){
        let div =  document.createElement("div");
        div.setAttribute("style","float:left");
        main.appendChild(div);

        counters.push([]);
        fractions.push([]);
        inside.push([]);

        let tab = document.createElement("table");
        tab.setAttribute("class","mtable");
        tab.setAttribute("style","margin-left: 0.5em");
        generateTable(tab, counters[i], fractions[i], inside[i]);
        generateTableHead(tab, "FEB "+i);

        div.appendChild(tab);
    }

    mjsonrpc_db_get_values(["/Equipment/Switching/Variables/SCSO"]).then(function(rpc) {
        update_scso(rpc.result.data[0]);
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

