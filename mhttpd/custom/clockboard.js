var canvas = document.querySelector('canvas');

var c = canvas.getContext('2d');


function Motherboard(x,y,dx,dy){
	this.x = x;
	this.y = y;
    this.dx= dx;
	this.dy= dy;

    this.current = "123 mA";
    this.voltage = "4567 mV";
    this.fans = "Fans: 0 mV";

    this.rx = new RXFirefly(this.x+500, this.y+130,140, 140);
    this.txclk = new TXFirefly(this.x+300, this.y+30,140, 140, "TXClk");
    this.txrst = new TXFirefly(this.x+300, this.y+200,140, 140, "TXRst");

	this.draw = function(){
        c.fillStyle = "rgba(0,0,0,0.5)";
        c.fillRect(this.x+10, this.y+10,this.dx, this.dy);
		c.fillStyle = "rgb(60,180,60)";
		c.fillRect(this.x, this.y,this.dx, this.dy);

        c.fillStyle = "Black";
        c.font = "12px Arial, sans-serif";
        c.fillText(this.current, this.x+40, this.y+100);
        c.fillText(this.voltage, this.x+100, this.y+100);
        c.fillText(this.fans, this.x+40, this.y+200);

        this.rx.draw();
        this.txclk.draw();
        this.txrst.draw();
	}
}

function RXFirefly(x,y,dx,dy){
    this.x = x;
    this.y = y;
    this.dx= dx;
    this.dy= dy;

    this.voltage ="2500 mV";
    this.temp ="99 C";

    this.los = [12];
    this.channelmask = [12];

    this.draw = function(){
        c.fillStyle = "Black";
        c.fillRect(this.x, this.y,this.dx, this.dy);

        c.fillStyle = "White";
        c.font = "20px Arial, sans-serif";
        c.fillText("RX", this.x+5, this.y+130);
        c.font = "12px Arial, sans-serif";
        c.fillText(this.voltage, this.x+90, this.y+110);
        c.fillText(this.temp, this.x+90, this.y+130);

        c.font = "9px Arial, sans-serif";
        c.fillText("LOS", this.x+5, this.y+10);
        for(var i=0; i < 12; i++){
            if(this.los[i])
                c.fillStyle = "Red";
            else
                c.fillStyle = "Green";

            c.beginPath();
            c.arc(this.x+10, this.y+20+7*i, 2, 0, Math.PI*2, false);
            c.fill();

            if(this.channelmask[i])
                c.fillStyle = "Grey";
            else
                c.fillStyle = "White";
            c.beginPath();
            c.arc(this.x+15, this.y+20+7*i, 2, 0, Math.PI*2, false);
            c.fill();
        }
    }
}


function TXFirefly(x,y,dx,dy, name){
    this.x = x;
    this.y = y;
    this.dx= dx;
    this.dy= dy;
    this.name = name;

    this.voltage ="2500mV";
    this.temp ="99C";

    this.los = [12];
    this.channelmask = [12];
    this.invertmask = [12];

    this.draw = function(){
        c.fillStyle = "Black";
        c.fillRect(this.x, this.y,this.dx, this.dy);

        c.fillStyle = "White";
        c.font = "20px Arial, sans-serif";
        c.fillText(this.name, this.x+5, this.y+130);
        c.font = "12px Arial, sans-serif";
        c.fillText(this.voltage, this.x+90, this.y+110);
        c.fillText(this.temp, this.x+90, this.y+130);

        c.font = "8px Arial, sans-serif";
        c.fillText("L A I", this.x+5, this.y+10);
        for(var i=0; i < 12; i++){
            if(this.los[i])
                c.fillStyle = "Red";
            else
                c.fillStyle = "Green";

            c.beginPath();
            c.arc(this.x+10, this.y+20+7*i, 2, 0, Math.PI*2, false);
            c.fill();

            if(this.channelmask[i])
                c.fillStyle = "Grey";
            else
                c.fillStyle = "White";

            c.beginPath();
            c.arc(this.x+15, this.y+20+7*i, 2, 0, Math.PI*2, false);
            c.fill();

            if(this.invertmask[i])
                c.fillStyle = "Blue";
            else
                c.fillStyle = "White";
            c.beginPath();
            c.arc(this.x+20, this.y+20+7*i, 2, 0, Math.PI*2, false);
            c.fill();
        }
    }
}



var motherboard = new Motherboard(50,200,700,400)

function Daughterboard(x,y,dx,dy, index){
	this.x = x;
	this.y = y;
	this.dx= dx;
	this.dy= dy;
	this.index = index;
	this.active = true;

	this.current = "123 mA";
	this.voltage = "4567 mV";

	this.draw = function(){
		if(this.active){
            c.fillStyle = "rgba(0,0,0,0.5)";
            c.fillRect(this.x+5, this.y+5,this.dx, this.dy);
			c.fillStyle = "rgb(100,200,100)"
			c.fillRect( this.x, this.y, this.dx, this.dy);
			c.fillStyle = "Black";
			c.font = "20px Arial, sans-serif";
			c.fillText(index, this.x+5, this.y+this.dy-5);
			c.fillStyle = "Black";
			c.font = "12px Arial, sans-serif";
			c.fillText(this.current, this.x+40, this.y+this.dy-5);
			c.fillText(this.voltage, this.x+100, this.y+this.dy-5);
		}
	}
}

var daughterboards = [];

function Firefly(x,y,dx,dy, daughter, index){
	this.x = x;
	this.y = y;
	this.dx= dx;
	this.dy= dy;
	this.daughter = daughter;
	this.index = index;
	this.active = true;
	this.lx = x-10;
	this.ly = y-10;
    this.ldx= 140;
    this.ldy= 140;

	this.ok= true;
    this.voltage ="2500mV";
    this.temp ="99C";
    this.tempcolour = "Black";

    this.los = [12];
    this.channelmask = [12];
    this.invertmask = [12];

	this.draw = function(){
		if(this.active){
			c.fillStyle = "rgb(90,90,90)";
			c.fillRect( this.x, this.y, this.dx, this.dy);
            if(this.ok && this.tempcolour != "Red")
				c.fillStyle = "Green";
			else
				c.fillStyle = "Red";
			c.beginPath();
			c.arc(this.x+20, this.y+20, 10, 0, Math.PI*2, false);
			c.fill();

            if(this.tempcolour != "Black"){
                c.fillStyle = "rgb(0,0,0)";
                c.fillRect( this.x, this.y+dy, this.dx, 15);
            }
            c.fillStyle = this.tempcolour;
            c.font = "12px Arial, sans-serif";
            c.fillText(this.temp, this.x+10, this.y+this.dy+12);
		}
	}

	this.drawLarge = function(){
		if(this.active){
			c.fillStyle = "rgb(20,20,20)";
			c.fillRect( this.lx, this.ly, this.ldx, this.ldy);
            c.fillStyle = "White";
            c.font = "20px Arial, sans-serif";
            c.fillText(this.daughter+"-"+this.index, this.lx+5, this.ly+130);
            c.font = "12px Arial, sans-serif";
            c.fillText(this.voltage, this.lx+90, this.ly+110);
            c.fillText(this.temp, this.lx+90, this.ly+130);

            c.font = "8px Arial, sans-serif";
            c.fillText("L A I", this.lx+5, this.ly+10);
            for(var i=0; i < 12; i++){
                if(this.los[i])
                    c.fillStyle = "Red";
                else
                    c.fillStyle = "Green";

                c.beginPath();
                c.arc(this.lx+10, this.ly+20+7*i, 2, 0, Math.PI*2, false);
                c.fill();

                if(this.channelmask[i])
                    c.fillStyle = "Grey";
                else
                    c.fillStyle = "White";

                c.beginPath();
                c.arc(this.lx+15, this.ly+20+7*i, 2, 0, Math.PI*2, false);
                c.fill();

                if(this.invertmask[i])
                    c.fillStyle = "Blue";
                else
                    c.fillStyle = "White";
                c.beginPath();
                c.arc(this.lx+20, this.ly+20+7*i, 2, 0, Math.PI*2, false);
                c.fill();
            }
		}
	}	
}

var fireflys = [];


function init(){

    //console.log("Init!");

	for(var i=0; i < 8; i++){
		var xpos = 25+i%4*180;
        var ypos = 25+Math.floor(i/4)*550;
		var xwidth = 160;
		var yheight = 200;
		daughterboards.push(new Daughterboard(xpos, ypos, xwidth, yheight, i));
		for(var j=0; j < 3; j++){
			var fxpos = xpos + 10 + j*50;
			var fypos = ypos + 10;
			var fxwidth = 40;
			var fyheight = 40;
			fireflys.push(new Firefly(fxpos, fypos, fxwidth, fyheight, i, j));
		}
	}
    mjsonrpc_db_get_values(["/Equipment/Clock Reset/Settings"]).then(function(rpc) {
       update_masks(rpc.result.data[0]);
    }).catch(function(error) {
       mjsonrpc_error_alert(error);
    });

}

var selindex = -1;

function draw(selindex){
    //console.log("Drawing!");

    canvas = document.querySelector('canvas');
    c = canvas.getContext('2d');

	c.fillStyle = "Silver";
	c.fillRect(0, 0, canvas.width, canvas.height);

	motherboard.draw();

	for(var i=0; i < 8; i++){
		daughterboards[i].draw();
        if(daughterboards[i].active){
            for(var j=0; j < 3; j++){
                fireflys[i*3+j].draw();
            }
        }
	}
	if(selindex >= 0){
		fireflys[selindex].drawLarge();
	}
}

init();
//draw(selindex);

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
		for(var j=0; j < 3; j++){
			if(fireflys[i*3+j].active && !found){
				if(selindex == i*3+j){
					if(mouse.x >= fireflys[i*3+j].lx &&
						mouse.x <= fireflys[i*3+j].lx +fireflys[i*3+j].ldx &&
						mouse.y >= fireflys[i*3+j].ly &&
						mouse.y <= fireflys[i*3+j].ly + fireflys[i*3+j].ldy){
							selindex = -1;
							found = true;
						}
				} else {
					if(mouse.x >= fireflys[i*3+j].x &&
						mouse.x <= fireflys[i*3+j].x +fireflys[i*3+j].dx &&
						mouse.y >= fireflys[i*3+j].y &&
						mouse.y <= fireflys[i*3+j].y + fireflys[i*3+j].dy){
							selindex = i*3+j;
							found = true;
					}
				}
			}		
		}
	}
	if(found)
		draw(selindex);
})


function update_boarddrawing(valuex) {

    var value = valuex;
    if(typeof valuex === 'string')
        value = JSON.parse(valuex);

    //console.log("Update BD!");


    var doffset = 12;
    var dnum = 11;

    motherboard.current = Number.parseFloat(value["crt1"][0]).toFixed(0) + "mA";
    motherboard.voltage = Number.parseFloat(value["crt1"][1]).toFixed(0) + "mV";

    for(var i=0; i < 12; i++){
        motherboard.rx.los[i] = value["crt1"][2] & (1<<i);
    }
    motherboard.rx.temp = Number.parseFloat(value["crt1"][3]).toFixed(0) + "C";
    motherboard.rx.voltage = Number.parseFloat(value["crt1"][4]).toFixed(0) + "mV";

    for(var i=0; i < 12; i++){
        motherboard.txclk.los[i] = value["crt1"][5] & (1<<i);
    }
    motherboard.txclk.temp = Number.parseFloat(value["crt1"][6]).toFixed(0) + "C";
    motherboard.txclk.voltage = Number.parseFloat(value["crt1"][7]).toFixed(0) + "mV";

    for(var i=0; i < 12; i++){
        motherboard.txrst.los[i] = value["crt1"][8] & (1<<i);
    }
    motherboard.txrst.temp = Number.parseFloat(value["crt1"][9]).toFixed(0) + "C";
    motherboard.txrst.voltage = Number.parseFloat(value["crt1"][10]).toFixed(0) + "mV";

    motherboard.fans = "Fan current: " + Number.parseFloat(value["crt1"][11].toFixed(0)) + "mA";



    var dp = value["daughters present"];
    var fp = value["fireflys present"];
    for(var i=0; i < 8; i++){
        if((dp & (1<<i)) > 0){
            daughterboards[i].active =  true;
            for(var j=0; j < 3; j++){
                 if((fp[i] & (1<<j)) > 0){
                     fireflys[i*3+j].active = true;
                     fireflys[i*3+j].ok = (value["crt1"][doffset+i*dnum+j*3+2] == 0);
                     for(var k=0; k < 12; k++){
                         fireflys[i*3+j].los[k] = value["crt1"][doffset+i*dnum+j*3+2] & (1<<k);
                     }
                     var t = value["crt1"][doffset+i*dnum+j*3+3];
                     fireflys[i*3+j].temp = Number.parseFloat(t).toFixed(0) + "C";
                     if(t < 55)
                         fireflys[i*3+j].tempcolour = "Black";
                     if(t >= 55)
                         fireflys[i*3+j].tempcolour = "Orange";
                     if(t >= 65)
                         fireflys[i*3+j].tempcolour = "Red";
                     fireflys[i*3+j].voltage = Number.parseFloat(value["crt1"][doffset+i*dnum+j*3+4]).toFixed(0) + "mV";
                 }  else {
                     fireflys[i*3+j].active = false;
                 }
            }
            daughterboards[i].current = Number.parseFloat(value["crt1"][doffset+i*dnum]).toFixed(0) + "mA";
            daughterboards[i].voltage = Number.parseFloat(value["crt1"][doffset+i*dnum+1]).toFixed(0) + "mV";
         } else {
            daughterboards[i].active =  false;
             for(var j=0; j < 3; j++){
                  fireflys[i*3+j].active = false;
             }
        }
    }

    draw(selindex);
}

function update_masks(value) {
    //console.log("Update M!");

    var rxmask = value["rx_mask"];
    for(var i=0; i < 12; i++){
        motherboard.rx.channelmask[i] = ((rxmask & (1<<i)) != 0);
    }

    var txclkmask = value["tx_clk_mask"];
    for(var i=0; i < 12; i++){
        motherboard.txclk.channelmask[i] = ((txclkmask & (1<<i)) != 0);
    }

    var txrstmask = value["tx_rst_mask"];
    for(var i=0; i < 12; i++){
        motherboard.txrst.channelmask[i] = ((txrstmask & (1<<i)) != 0);
    }

    var txclkinvmask = value["tx_clk_invert_mask"];
    for(var i=0; i < 12; i++){
        motherboard.txclk.invertmask[i] = ((txclkinvmask & (1<<i)) != 0);
    }

    var txrstinvmask = value["tx_rst_invert_mask"];
    for(var i=0; i < 12; i++){
        motherboard.txrst.invertmask[i] = ((txrstinvmask & (1<<i)) != 0);
    }

    var txmask = value["tx_mask"];
    var txinvmask = value["tx_invert_mask"];
    for(var i=0; i < 8; i++){
        for(var j=0; j < 3; j++){
            for(var k=0; k < 12; k++){
               fireflys[i*3+j].channelmask[k] = ((txmask & (1<<k)) != 0);
               fireflys[i*3+j].invertmask[k] = ((txinvmask & (1<<k)) != 0);
            }
         }
    }

//    console.log(value + " " + motherboard.rx.channelmask)
    draw(selindex);
}
