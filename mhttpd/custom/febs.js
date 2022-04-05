var canvas = document.querySelector('canvas');
var cc = canvas.getContext('2d');

var boardselindex = 0;
var nfebs = [34,33,33,12];

var xoffset = 300;
var yoffset = 25;

function write_temperature(name, temp, x, y){
    var tempcolour;
    if(temp < 55)
        tempcolour = "Black";
    if(temp >= 55)
        tempcolour = "Orange";
    if(temp >= 65)
        tempcolour = "Red";

    cc.fillStyle = tempcolour;
    cc.font = "12px Arial, sans-serif";
    cc.textAlign = "right";
    cc.fillText(name + " " + temp.toFixed(0)  + " C", x, y);
    cc.textAlign = "left";
}

function write_voltage(name, volt, x, y){
    cc.fillStyle = "Black";
    cc.font = "12px Arial, sans-serif";
    cc.textAlign = "right";
    cc.fillText(name + " " + volt.toFixed(2)  + " V", x, y);
    cc.textAlign = "left";
}

function write_power(name, power, x, y){
    cc.fillStyle = "Black";
    cc.font = "12px Arial, sans-serif";
    cc.textAlign = "right";
    power = power * 1e6;
    cc.fillText(name + " " + power.toFixed(0)  + " muW", x, y);
    cc.textAlign = "left";
}

function tbar(temp, x, y, dy){
    var tempcolour = "Blue";
    if(temp >= 40)
        tempcolour = "Green";
    if(temp >= 55)
        tempcolour = "Orange";
    if(temp >= 65)
        tempcolour = "Red";

    var tempfrac = temp/100.0;
    if(tempfrac < 0.1)
        tempfrac = 0.1;
    if(tempfrac > 1)
        tempfrac = 1;

    cc.beginPath();
    cc.strokeStyle = tempcolour;
    cc.moveTo(x,y+dy);
    cc.lineTo(x,y+dy-dy*tempfrac);
    cc.stroke();

}

function rxdot(power,x,y){
    var rxcol = "Red";
    power = power*1e6;
    if(power > 100)
        var rxcol = "Orange";
    if(power > 200)
        var rxcol = "Blue";

    cc.fillStyle = rxcol;
    cc.beginPath();
    cc.arc(x, y, 2, 0, Math.PI*2, false);
    cc.fill();
}


function Feb(x,y,dx,dy, index){
    this.x = x;
    this.y = y;
    this.dx= dx;
    this.dy= dy;

    this.index = index;
    this.febnum = 0;
    this.active = 0;
    this.name = "";
    this.type = "";
    this.shorttype = "";

    //Crate and slot
    this.crate = -1;
    this.slot = -1;

    //Firmware versions
    this.arria_firmware =0;
    this.max_firmware =0;

    //Here comes the content of the slow control bank
    this.arria_temp = 0;
    this.max_temp   = 0;
    this.si1_temp   = 0;
    this.si2_temp   = 0;
    this.arria_temp_ext = 0;
    this.dcdc_temp      = 0;
    this.v1_1           = 0;
    this.v1_8           = 0;
    this.v2_5           = 0;
    this.v3_3           = 0;
    this.v20            = 0;
    this.ff1_temp       = 0;
    this.ff1_volt       = 0;
    this.ff1_rx1        = 0;
    this.ff1_rx2        = 0;
    this.ff1_rx3        = 0;
    this.ff1_rx4        = 0;
    this.ff1_alarms     = 0;

    this.draw = function(sel){
        if(sel){
           cc.fillStyle = "rgb(0,0,200)";
           cc.fillRect(this.x-2, this.y-2,this.dx+4, this.dy+4);
           this.drawbig();
        }
        if(this.active > 0)
            cc.fillStyle = "rgb(150,255,150)";
        else
            cc.fillStyle = "rgb(200,200,200)";
        cc.fillRect(this.x, this.y,this.dx, this.dy);

        cc.fillStyle = "Black";
        cc.font = "10px Arial, sans-serif";
        cc.fillText(this.index, this.x+2, this.y+10);

        cc.fillStyle = "Black";
        cc.font = "10px Arial, sans-serif";
        cc.fillText(this.crate + "/" + this.slot, this.x+2, this.y+20);

        cc.fillStyle = "Black";
        cc.font = "10px Arial, sans-serif";
        cc.fillText(this.shorttype, this.x+this.dx-10, this.y+10);

        if(this.active > 0){
            tbar(this.arria_temp,this.x+50, this.y+2, this.dy-4);
            tbar(this.max_temp  ,this.x+54, this.y+2, this.dy-4);
            tbar(this.si1_temp  ,this.x+58, this.y+2, this.dy-4);
            tbar(this.si2_temp  ,this.x+62, this.y+2, this.dy-4);
            tbar(this.arria_temp_ext,this.x+66, this.y+2, this.dy-4);
            tbar(this.dcdc_temp     ,this.x+70, this.y+2, this.dy-4);
            tbar(this.ff1_temp      ,this.x+74, this.y+2, this.dy-4);

            rxdot(this.ff1_rx1, this.x+80, this.y+6);
            //rxdot(this.ff1_rx2, this.x+86, this.y+6);
            rxdot(this.ff1_rx3, this.x+80, this.y+12);
            rxdot(this.ff1_rx4, this.x+86, this.y+12);

            cc.fillStyle = "Black";
            cc.font = "10px Arial, sans-serif";
            cc.fillText(this.arria_firmware.toString(16), this.x+this.dx-70, this.y+10);
            cc.fillText(this.max_firmware.toString(16), this.x+this.dx-70, this.y+20);

        }

    }

    this.drawbig = function(){
        cc.fillStyle = "rgb(150,255,150)";
        var xstart = 10+xoffset*3;
        var ystart = 40+yoffset*14;
        var xwidth = 280;
        cc.fillRect(xstart, ystart, xwidth, 400);

        cc.fillStyle = "Black";
        cc.font = "12px Arial, sans-serif";
        cc.fillText(this.index, xstart+2, ystart+12);

        cc.fillStyle = "Black";
        cc.font = "12px Arial, sans-serif";
        cc.fillText("Crate: " + this.crate + " - Slot: " + this.slot, xstart+xwidth/2-50, ystart+12);

        cc.fillStyle = "Black";
        cc.font = "12px Arial, sans-serif";
        cc.fillText(this.type, xstart+xwidth-40, ystart+12);

        cc.fillStyle = "Black";
        cc.font = "14px Arial, sans-serif";
        cc.fillText(this.name, xstart+20, ystart+40);

        cc.fillStyle = "Black";
        cc.font = "12px Arial, sans-serif";
        cc.fillText("Arria firmware:    "+this.arria_firmware.toString(16),
                    xstart+10, ystart+170);
        cc.fillStyle = "Black";
        cc.font = "12px Arial, sans-serif";
        cc.fillText("Max10 firmware: "+this.max_firmware.toString(16),
                    xstart+10, ystart+185);

        write_temperature("Arria", this.arria_temp, xstart+100,ystart+60);
        write_temperature("Max",   this.max_temp,   xstart+100,ystart+75);
        write_temperature("Si1",   this.si1_temp,   xstart+100,ystart+90);
        write_temperature("Si2",   this.si2_temp,   xstart+100,ystart+105);
        write_temperature("Arria ext",   this.arria_temp_ext,   xstart+100,ystart+120);
        write_temperature("DCDC",   this.arria_temp_ext,   xstart+100,ystart+135);
        write_temperature("Firefly 1",   this.ff1_temp,   xstart+100,ystart+150);

        write_voltage("1.1V", this.v1_1, xstart+200, ystart+60);
        write_voltage("1.8V", this.v1_8, xstart+200, ystart+75);
        write_voltage("2.5V", this.v2_5, xstart+200, ystart+90);
        write_voltage("3.3V", this.v3_3, xstart+200, ystart+105);
        write_voltage("20V",  this.v20,  xstart+200, ystart+120);
        write_voltage("FF",   this.ff1_volt,  xstart+200, ystart+135);

        write_power("RX1", this.ff1_rx1, xstart+260, ystart+170);
        write_power("RX2", this.ff1_rx2, xstart+260, ystart+185);
        write_power("RX3", this.ff1_rx3, xstart+260, ystart+200);
        write_power("RX4", this.ff1_rx4, xstart+260, ystart+215);


    }
}

function Switchingboard(index){
    this.index = index;
    this.active = 0;
    this.name = "";
}

var switchingboards = new Array(4);
var febs = new Array(4);
for(var i=0; i < 4; i++){
    febs[i] = new Array(34);
    switchingboards[i] = new Switchingboard(i);
    for(var j=0; j < nfebs[i]; j++){
        febs[i][j] = new Feb(10+xoffset*i, 40+yoffset*j, 280, 23, i*48+j);
    }
}





function init(){

    mjsonrpc_db_get_values(["/Equipment/Clock Reset/Settings"]).then(function(rpc) {
        update_swmask(rpc.result.data[0],0);
     }).catch(function(error) {
        mjsonrpc_error_alert(error);
     });

    mjsonrpc_db_get_values(["/Equipment/SwitchingCentral/Variables"]).then(function(rpc) {
        if(rpc.result.data[0]){
            var scvals = rpc.result.data[0]["SCFE"];
            if(scvals)
                 update_sc(scvals, 0);
        }
    }).catch(function(error) {
        mjsonrpc_error_alert(error);
    });    
     
    mjsonrpc_db_get_values(["/Equipment/SwitchingUpstream/Variables"]).then(function(rpc) {
        if(rpc.result.data[0]){
            var scvals = rpc.result.data[0]["SUFE"];
            if(scvals)
                update_sc(scvals, 1);
        }
    }).catch(function(error) {
                mjsonrpc_error_alert(error);
    });    
          
    mjsonrpc_db_get_values(["/Equipment/SwitchingDownstream/Variables"]).then(function(rpc) {
        if(rpc.result.data[0]){
            var scvals = rpc.result.data[0]["SDFE"];
            if(scvals)
                update_sc(scvals, 2);
        }
    }).catch(function(error) {
                mjsonrpc_error_alert(error);
    });         
    
    mjsonrpc_db_get_values(["/Equipment/SwitchingFibres/Variables"]).then(function(rpc) {
        if(rpc.result.data[0]){
            var scvals = rpc.result.data[0]["SFFE"];
            if(scvals)
                update_sc(scvals, 3);
        }
    }).catch(function(error) {
                mjsonrpc_error_alert(error);
    });      
    
    draw(-1);
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

    for(var i=0; i < 4; i++){
        for(var j=0; j < nfebs[i]; j++){
            if(febs[i][j].active && !found){
                if(mouse.x >= febs[i][j].x &&
                   mouse.x <= febs[i][j].x +febs[i][j].dx &&
                   mouse.y >= febs[i][j].y &&
                   mouse.y <= febs[i][j].y + febs[i][j].dy){
                        selindex = i*48+j;
                        found = true;
                        break;
                }
            }
        }
        if(found) break;
    }
    if(found){
        boardselindex = selindex;
        draw(selindex);
    }
})


function draw(boardselindex){
    canvas = document.querySelector('canvas');
    cc = canvas.getContext('2d');

    cc.fillStyle = "Silver";
    cc.fillRect(0, 0, canvas.width, canvas.height);

    for(var i=0; i < 4; i++){

        cc.fillStyle = "Black";
        cc.font = "14px Arial, sans-serif";
        cc.fillText(switchingboards[i].name, 40 + xoffset*i, 20);

        for(var j=0; j < nfebs[i]; j++){
            if(Math.floor(boardselindex/48) == i && boardselindex%48 == j)
                febs[i][j].draw(1);
            else
                febs[i][j].draw(0);
        }
    }
}

init();

function update_swmask(valuex) {
    var value = valuex;
    if(typeof valuex === 'string')
        value = JSON.parse(valuex);
    var swmask = value["switchingboardmask"];
    var swnames = value["switchingboardnames"];

    for(var i=0; i < 4; i++){
        switchingboards[i].active = swmask[i];
        switchingboards[i].name = swnames[i];
    }

    if(switchingboards[0].active){
        mjsonrpc_db_get_values(["/Equipment/LinksCentral/Settings"]).then(function(rpc) {
            update_masks(rpc.result.data[0],0);
        }).catch(function(error) {
            mjsonrpc_error_alert(error);
        });

        mjsonrpc_db_get_values(["/Equipment/LinksCentral/Variables/FEBFirmware"]).then(function(rpc) {
            update_firmware(rpc.result.data[0],0);
         }).catch(function(error) {
            mjsonrpc_error_alert(error);
         });
    }

    if(switchingboards[1].active){
        mjsonrpc_db_get_values(["/Equipment/LinksUpstream/Settings"]).then(function(rpc) {
            update_masks(rpc.result.data[0],1);
        }).catch(function(error) {
            mjsonrpc_error_alert(error);
        });

        mjsonrpc_db_get_values(["/Equipment/LinksUpstream/Variables/FEBFirmware"]).then(function(rpc) {
            update_firmware(rpc.result.data[0],1);
         }).catch(function(error) {
            mjsonrpc_error_alert(error);
         });
    }  
    
    if(switchingboards[2].active){
        mjsonrpc_db_get_values(["/Equipment/LinksDownstream/Settings"]).then(function(rpc) {
            update_masks(rpc.result.data[0],2);
         }).catch(function(error) {
            mjsonrpc_error_alert(error);
        });

        mjsonrpc_db_get_values(["/Equipment/LinksDownstream/Variables/FEBFirmware"]).then(function(rpc) {
            update_firmware(rpc.result.data[0],3);
         }).catch(function(error) {
            mjsonrpc_error_alert(error);
         });
    }

    if(switchingboards[3].active){
        mjsonrpc_db_get_values(["/Equipment/LinksFibre/Settings"]).then(function(rpc) {
            update_masks(rpc.result.data[0],3);
        }).catch(function(error) {
            mjsonrpc_error_alert(error);
        });

        mjsonrpc_db_get_values(["/Equipment/LinksFibre/Variables/FEBFirmware"]).then(function(rpc) {
            update_firmware(rpc.result.data[0],4);
        }).catch(function(error) {
            mjsonrpc_error_alert(error);
        });
    }

    draw(boardselindex);
}


function update_masks(valuex, swb) {

    var value = valuex;
    if(typeof valuex === 'string')
        value = JSON.parse(valuex);

    var femask = value["linkmask"];
    var linkfeb = value["linkfeb"];
    var fenames = value["febname"];
    var fetypes = value["febtype"];

    for(var j=0; j < nfebs[swb]; j++){
        febs[swb][j].name     = fenames[j];
        febs[swb][j].active   = femask[j];
        febs[swb][j].febnum   = parseInt(linkfeb[j],16);
        if(fetypes[j] == 1){
            febs[swb][j].type = "Pixel";
            febs[swb][j].shorttype = "P";
        } else if(fetypes[j] == 2){
            febs[swb][j].type = "Fibre";
            febs[swb][j].shorttype = "F";
        } else if   (fetypes[swb] == 3){
            febs[swb][j].type = "Tiles";
            febs[swb][j].shorttype = "T";
        } else if   (fetypes[swb] == 4){
            febs[swb][j].type = "FibreSecondary";
            febs[swb][j].shorttype = "FS";
            febs[swb][j].active   = 0;
        } else {
            febs[swb][j].type = "Undef.";
            febs[swb][j].shorttype = "U";    
        }
    }

    mjsonrpc_db_get_values(["/Equipment/FEBCrates/Settings"]).then(function(rpc) {
        update_slots(rpc.result.data[0]);
     }).catch(function(error) {
        mjsonrpc_error_alert(error);
     });


    draw(boardselindex);
}

function update_firmware(valuex, swb) {

    var value = valuex;
    if(typeof valuex === 'string')
        value = JSON.parse(valuex);

    var arria = value["arria v firmware version"];
    var max = value["max 10 firmware version"];

    for(var j=0; j < nfebs[swb]; j++){
        febs[swb][j].arria_firmware = arria[j];
        febs[swb][j].max_firmware   = max[j];    
    }
    draw(boardselindex);
}

function update_sc(valuex, swindex) {

    var value = valuex;
    if(typeof valuex === 'string')
        value = JSON.parse(valuex);


    for(var j=0; j < nfebs[swindex]; j++){
        febs[swindex][j].arria_temp     = value[j*26+1];
        febs[swindex][j].max_temp       = value[j*26+2];
        febs[swindex][j].si1_temp       = value[j*26+3];
        febs[swindex][j].si2_temp       = value[j*26+4];
        febs[swindex][j].arria_temp_ext = value[j*26+5];
        febs[swindex][j].dcdc_temp      = value[j*26+6];
        febs[swindex][j].v1_1           = value[j*26+7];
        febs[swindex][j].v1_8           = value[j*26+8];
        febs[swindex][j].v2_5           = value[j*26+9];
        febs[swindex][j].v3_3           = value[j*26+10];
        febs[swindex][j].v20            = value[j*26+11];
        febs[swindex][j].ff1_temp       = value[j*26+12];
        febs[swindex][j].ff1_volt       = value[j*26+13];
        febs[swindex][j].ff1_rx1        = value[j*26+14];
        febs[swindex][j].ff1_rx2        = value[j*26+15];
        febs[swindex][j].ff1_rx3        = value[j*26+16];
        febs[swindex][j].ff1_rx4        = value[j*26+17];
        febs[swindex][j].ff1_alarms     = value[j*26+18];
    }
    draw(boardselindex);
}

function update_slots(valuex){
    var value = valuex;
    if(typeof valuex === 'string')
        value = JSON.parse(valuex);

    var crate = value["febcrate"];
    var slot = value["febslot"];

    for(var i=0; i < 4; i++){
        for(var j=0; j < nfebs[i]; j++){
            var index = febs[i][j].febnum;
            if(index < 128){
                febs[i][j].crate = parseInt(crate[index],16);
                febs[i][j].slot  = parseInt(slot[index],16);
            }
        }
    }
    draw(boardselindex);
}

function start_programming() {
    if (confirm("Do you want to program this FEB?")) {
        mjsonrpc_db_set_value("/Equipment/SwitchingCentral/Commands/Load Firmware", 1);
    }
}
