var canvas = document.querySelector('canvas');
var cc = canvas.getContext('2d');

function write_temperature(name, temp, x, y){
    var tempcolour;
    if(temp < 55)
        tempcolour = "Black";
    if(temp >= 55)
        tempcolour = "Orange";
    if(temp >= 65)
        tempcolour = "Red";

    cc.fillStyle = tempcolour;
    cc.font = "12px Arial";
    cc.textAlign = "right";
    cc.fillText(name + " " + temp.toFixed(0)  + " C", x, y);
    cc.textAlign = "left";
}

function write_voltage(name, volt, x, y){
    cc.fillStyle = "Black";
    cc.font = "12px Arial";
    cc.textAlign = "right";
    cc.fillText(name + " " + volt.toFixed(2)  + " V", x, y);
    cc.textAlign = "left";
}

function FEB(x,y,dx,dy, slot){
    this.x = x;
    this.y = y;
    this.dx= dx;
    this.dy= dy;

    this.slot = slot;
    this.index = -1;
    this.powered = 0;

    this.draw = function(){
        if(this.powered > 0)
            cc.fillStyle = "rgb(80,80,180)";
        else
            cc.fillStyle = "rgb(200,200,200)";
        cc.fillRect(this.x, this.y,this.dx, this.dy);

        cc.fillStyle = "Black";
        cc.font = "12px Arial";
        cc.fillText("Slot " + this.slot, this.x+10, this.y+15);

        if(this.index >= 0){
            cc.fillStyle = "Black";
            cc.font = "12px Arial";
            cc.fillText("FEB " + this.index, this.x+10, this.y+30);
        }


        var text;
        if(this.powered > 0){
            text = "ON";
            cc.fillStyle = "rgb(0,180,0)";
        } else {
            text = "OFF";
            cc.fillStyle = "rgb(180,180,180)";
        }
        cc.fillRect(this.x+5, this.y+50,this.dx-10, this.dy-55);
        cc.fillStyle = "Black";
        cc.font = "15px Arial";
        cc.fillText(text, this.x+15, this.y+70);
    }

}

function Crate(x,y,dx,dy, index){
    this.x = x;
    this.y = y;
    this.dx= dx;
    this.dy= dy;

    this.index = index;
    this.active = 0;

    this.mscb_node = "mscb338";
    this.mscb_device = "15";
    this.U24 = 0.0;
    this.U33 = 0.0;
    this.U5  = 0.0;
    this.temp = 0.0;

    this.FEBs = Array(16);
    for(var j=0; j < 16; j++){
        this.FEBs[j] = new FEB(this.x+200+j*60,this.y+5, 55, this.dy-10, j);
    }

    this.draw = function(){
        if(this.active > 0)
            cc.fillStyle = "rgb(150,255,150)";
        else
            cc.fillStyle = "rgb(200,200,200)";
        cc.fillRect(this.x, this.y,this.dx, this.dy);

        cc.fillStyle = "Black";
        cc.font = "20px Arial";
        cc.fillText("Crate " + this.index, this.x+10, this.y+20);

        cc.fillStyle = "Black";
        cc.font = "12px Arial";
        cc.fillText(this.mscb_node + " / " + this.mscb_device, this.x+90, this.y+20);

        write_voltage("20V", this.U24, this.x+70, this.y+40);
        write_voltage("3.3V", this.U33, this.x+70, this.y+60);
        write_voltage("5V", this.U5, this.x+70, this.y+80);

        write_temperature("T", this.temp, this.x+120, this.y+60);

        for(var j=0; j < 16; j++){
            this.FEBs[j].draw();
        }

    }
}


var crates = new Array(8);
for(var i=0; i < 8; i++){
    crates[i] = new Crate(10, 10+i*105, canvas.width-20, 95, i);
}



function init(){
    mjsonrpc_db_get_values(["/Equipment/FEBCrates/Settings"]).then(function(rpc) {
        update_slots(rpc.result.data[0]);
     }).catch(function(error) {
        mjsonrpc_error_alert(error);
     });

    mjsonrpc_db_get_values(["/Equipment/FEBCrates/Variables"]).then(function(rpc) {
        update_sc(rpc.result.data[0]);
     }).catch(function(error) {
        mjsonrpc_error_alert(error);
     });

    draw();
}


var mouse = {
    x: undefined,
    y: undefined
}

window.addEventListener('click', function(event) {
    var rect = canvas.getBoundingClientRect();

    mouse.x = event.clientX -rect.left;
    mouse.y = event.clientY -rect.top;

    var found = false;

    for(var i=0; i < 8; i++){
        if(crates[i].active === 0)
            continue;
        for(var j=0; j < 16; j++){
            var ind = crates[i].FEBs[j].index;
            if(!found && ind >= 0){
                if(mouse.x >= crates[i].FEBs[j].x &&
                   mouse.x <= crates[i].FEBs[j].x +crates[i].FEBs[j].dx &&
                   mouse.y >= crates[i].FEBs[j].y &&
                   mouse.y <= crates[i].FEBs[j].y + crates[i].FEBs[j].dy){
                        var oldval =     crates[i].FEBs[j].powered;
                        var newval = 0;
                        if(oldval === 0)
                            newval = 1 ;
                        mjsonrpc_db_set_value("/Equipment/FEBCrates/Variables/FEBPower["+ind+"]", newval);
                        crates[i].FEBs[j].powered = newval;
                        found = true;
                        break;
                }
            }
        }
        if(found) break;
    }
    if(found)
        draw();
})


function draw(){
    canvas = document.querySelector('canvas');
    cc = canvas.getContext('2d');

    cc.fillStyle = "Silver";
    cc.fillRect(0, 0, canvas.width, canvas.height);

    for(var i=0; i < 8; i++){
        crates[i].draw();
    }
}

init();

function update_sc(valuex){
    var value = valuex;
    if(typeof valuex === 'string')
        value = JSON.parse(valuex);
    var scfcpercrate = 21;
    var scfc = value["scfc"];
    for(var i=0; i < 8; i++){
        if(scfc[1+scfcpercrate*i] > 5){
           crates[i].active = 1;
            crates[i].U24 = scfc[1+scfcpercrate*i];
            crates[i].U33 = scfc[2+scfcpercrate*i];
            crates[i].U5  = scfc[3+scfcpercrate*i];
            crates[i].temp= scfc[4+scfcpercrate*i];

            for(var j=0; j < 16; j++){
                if(crates[i].FEBs[j].index >= 0)
                    crates[i].FEBs[j].powered = value["febpower"][crates[i].FEBs[j].index];
            }
        } else {
            crates[i].active = 0;
        }
    }

    draw();
}

function update_slots(valuex){
    var value = valuex;
    if(typeof valuex === 'string')
        value = JSON.parse(valuex);

    var cindex = value["febcrate"];
    var sindex = value["febslot"];


    for(var i=0; i < cindex.length; i++){
        var c = parseInt(cindex[i],16);
        var s = parseInt(sindex[i],16);
        if(c < 8 && s < 16){
            crates[c].FEBs[s].index = i;
        }
    }

    for(var i=0; i < 8; i++){
        crates[i].mscb_node = value["cratecontrollermscb"][i];
        crates[i].mscb_device = parseInt(value["cratecontrollernode"][i],16);
    }
}

