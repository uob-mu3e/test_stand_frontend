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
    cc.font = "12px Arial";
    cc.textAlign = "right";
    cc.fillText(name + " " + temp + " C", x, y);
    cc.textAlign = "left";
}

function Feb(x,y,dx,dy, index){
    this.x = x;
    this.y = y;
    this.dx= dx;
    this.dy= dy;

    this.index = index;
    this.active = 0;
    this.name = "";
    this.type = "";
    this.shorttype = "";

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
        cc.font = "10px Arial";
        cc.fillText(this.index, this.x+2, this.y+10);

        cc.fillStyle = "Black";
        cc.font = "10px Arial";
        cc.fillText(this.shorttype, this.x+this.dx-10, this.y+10);

    }

    this.drawbig = function(){
        cc.fillStyle = "rgb(150,255,150)";
        var xstart = 10+xoffset*3;
        var ystart = 40+yoffset*14;
        var xwidth = 280;
        cc.fillRect(xstart, ystart, xwidth, 400);

        cc.fillStyle = "Black";
        cc.font = "12px Arial";
        cc.fillText(this.index, xstart+2, ystart+12);

        cc.fillStyle = "Black";
        cc.font = "12px Arial";
        cc.fillText(this.type, xstart+xwidth-40, ystart+12);

        cc.fillStyle = "Black";
        cc.font = "14px Arial";
        cc.fillText(this.name, xstart+20, ystart+20);

        write_temperature("Arria", this.arria_temp, xstart+100,ystart+40);
        write_temperature("Max",   this.max_temp,   xstart+100,ystart+55);
        write_temperature("Si1",   this.si1_temp,   xstart+100,ystart+70);
        write_temperature("Si2",   this.si2_temp,   xstart+100,ystart+85);

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

   mjsonrpc_db_get_values(["/Equipment/Links/Settings"]).then(function(rpc) {
       update_masks(rpc.result.data[0]);
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
    if(found)
        draw(selindex);
})


function draw(boardselindex){
    canvas = document.querySelector('canvas');
    cc = canvas.getContext('2d');

    cc.fillStyle = "Silver";
    cc.fillRect(0, 0, canvas.width, canvas.height);

    for(var i=0; i < 4; i++){

        cc.fillStyle = "Black";
        cc.font = "14px Arial";
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


function update_masks(valuex) {

    var value = valuex;
    if(typeof valuex === 'string')
        value = JSON.parse(valuex);


    var swmask = value["switchingboardmask"];
    var swnames = value["switchingboardnames"];
    var femask = value["linkmask"];
    var fenames = value["frontendboardnames"];
    var fetypes = value["frontendboardtype"];

    for(var i=0; i < 4; i++){
        switchingboards[i].active = swmask[i];
        switchingboards[i].name  = swnames[i];
        for(var j=0; j < nfebs[i]; j++){
            var index = 48*i+j;
            febs[i][j].name     = fenames[index];
            febs[i][j].active   = femask[index];
            if(fetypes[index] == 1){
                febs[i][j].type = "Pixel";
                febs[i][j].shorttype = "P";
            } else if(fetypes[index] == 2){
                febs[i][j].type = "Fibre";
                febs[i][j].shorttype = "F";
            } else if   (fetypes[index] == 3){
                febs[i][j].type = "Tiles";
                febs[i][j].shorttype = "T";
            } else {
                febs[i][j].type = "Undefined Type";
                febs[i][j].shorttype = "U";
            }
        }
    }
    draw(boardselindex);
}

function update_sc(valuex) {

    var value = valuex;
    if(typeof valuex === 'string')
        value = JSON.parse(valuex);


    var scvals = value["SCFE"];
    var swindex = 0;
    if(!scvals){
        scvals = value["SUFE"];
        swindex = 1;
    }
    if(!scvals){
        scvals = value["SDFE"];
        swindex = 2;
    }
    if(!scvals){
        scvals = value["SFFE"];
        swindex = 3;
    }
    if(!scvals){
        console.log("SC info not found");
        return;
    }

    for(var j=0; j < nfebs[swindex]; j++){
        febs[swindex][j].arria_temp     = scvals[j*26+1];
        febs[swindex][j].max_temp       = scvals[j*26+2];
        febs[swindex][j].si1_temp       = scvals[j*26+3];
        febs[swindex][j].si2_temp       = scvals[j*26+4];
        febs[swindex][j].arria_temp_ext = scvals[j*26+5];
        febs[swindex][j].dcdc_temp      = scvals[j*26+6];
        febs[swindex][j].v1_1           = scvals[j*26+7];
        febs[swindex][j].v1_8           = scvals[j*26+8];
        febs[swindex][j].v2_5           = scvals[j*26+9];
        febs[swindex][j].v3_3           = scvals[j*26+10];
        febs[swindex][j].v20            = scvals[j*26+11];
        febs[swindex][j].ff1_temp       = scvals[j*26+12];
        febs[swindex][j].ff1_volt       = scvals[j*26+13];
        febs[swindex][j].ff1_rx1        = scvals[j*26+14];
        febs[swindex][j].ff1_rx2        = scvals[j*26+15];
        febs[swindex][j].ff1_rx3        = scvals[j*26+16];
        febs[swindex][j].ff1_rx4        = scvals[j*26+17];
        febs[swindex][j].ff1_alarms     = scvals[j*26+18];
    }
    draw(boardselindex);
}
