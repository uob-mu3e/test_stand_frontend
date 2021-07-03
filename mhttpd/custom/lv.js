function checkBox1(){
    var check = document.getElementById("Box1");
      //var text = document.getElementById("text");
      if (check.checked == true){
        modbset(["/Equipment/HAMEG1/Variables/Set State[0]", "/Equipment/HAMEG1/Variables/Set State[1]", "/Equipment/HAMEG1/Variables/Set State[2]", "/Equipment/HAMEG1/Variables/Set State[3]"],[1, 1, 1, 1]);
      } else {
        modbset(["/Equipment/HAMEG1/Variables/Set State[0]", "/Equipment/HAMEG1/Variables/Set State[1]", "/Equipment/HAMEG1/Variables/Set State[2]", "/Equipment/HAMEG1/Variables/Set State[3]"],[0, 0, 0, 0]);
      }
}

function setVolt1(){
    var VoltageInput = document.getElementById('inputVolt1').value;
modbset(["/Equipment/HAMEG1/Variables/Demand Voltage[0]", "/Equipment/HAMEG1/Variables/Demand Voltage[1]", "/Equipment/HAMEG1/Variables/Demand Voltage[2]", "/Equipment/HAMEG1/Variables/Demand Voltage[3]"],[VoltageInput, VoltageInput, VoltageInput, VoltageInput]);
}

function setCur1(){
    var CurrentInput = document.getElementById('inputCur1').value;
modbset(["/Equipment/HAMEG1/Variables/Current Limit[0]", "/Equipment/HAMEG1/Variables/Current Limit[1]", "/Equipment/HAMEG1/Variables/Current Limit[2]", "/Equipment/HAMEG1/Variables/Current Limit[3]"],[CurrentInput, CurrentInput, CurrentInput, CurrentInput]);
}
