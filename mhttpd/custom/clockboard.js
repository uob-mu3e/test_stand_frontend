var canvas = document.querySelector('canvas');

var c = canvas.getContext('2d');


function Motherboard(x,y,dx,dy){
	this.x = x;
	this.y = y;
	this.dx= dx;
	this.dy= dy;

    this.current = "123 mA";
    this.voltage = "4567 mV";

    this.rxffvoltage ="2500 mV";
    this.rxfftemp ="99 C";

    this.rxlos = [12];
    this.rxlosmask = [12];

	this.draw = function(){
        c.fillStyle = "rgba(0,0,0,0.5)";
        c.fillRect(this.x+10, this.y+10,this.dx, this.dy);
		c.fillStyle = "rgb(60,180,60)";
		c.fillRect(this.x, this.y,this.dx, this.dy);

        c.fillStyle = "Black";
        c.font = "12px Arial";
        c.fillText(this.current, this.x+40, this.y+100);
        c.fillText(this.voltage, this.x+100, this.y+100);

        c.fillStyle = "Black";
        c.fillRect(this.x+300, this.y+30,140, 140);

        c.fillStyle = "White";
        c.font = "20px Arial";
        c.fillText("RX", this.x+305, this.y+160);
        c.font = "12px Arial";
        c.fillText(this.rxffvoltage, this.x+345, this.y+160);
        c.fillText(this.rxfftemp, this.x+410, this.y+160);

        c.font = "9px Arial";
        c.fillText("LOS", this.x+305, this.y+40);
        for(var i=0; i < 12; i++){
            if(this.rxlos[i])
                c.fillStyle = "Red";
            else
                c.fillStyle = "Green";

            c.beginPath();
            c.arc(this.x+310, this.y+50+7*i, 2, 0, Math.PI*2, false);
            c.fill();

            if(this.rxlosmask[i])
                c.fillStyle = "Grey";
            else
                c.fillStyle = "White";
            c.beginPath();
            c.arc(this.x+315, this.y+50+7*i, 2, 0, Math.PI*2, false);
            c.fill();

        }
	}
}

var motherboard = new Motherboard(50,200,700,200)

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
			c.font = "20px Arial";
			c.fillText(index, this.x+5, this.y+this.dy-5);
			c.fillStyle = "Black";
			c.font = "12px Arial";
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
	this.ldx= dx+100;
	this.ldy= dy+100;

	this.ok= true;

	this.draw = function(){
		if(this.active){
			c.fillStyle = "rgb(90,90,90)";
			c.fillRect( this.x, this.y, this.dx, this.dy);
			if(this.ok)
				c.fillStyle = "Green";
			else
				c.fillStyle = "Red";
			c.beginPath();
			c.arc(this.x+20, this.y+20, 10, 0, Math.PI*2, false);
			c.fill();
		}
	}

	this.drawLarge = function(){
		if(this.active){
			c.fillStyle = "rgb(20,20,20)";
			c.fillRect( this.lx, this.ly, this.ldx, this.ldy);
			if(this.ok)
				c.fillStyle = "Green";
			else
				c.fillStyle = "Red";
			c.beginPath();
			c.arc(this.lx+20, this.ly+20, 10, 0, Math.PI*2, false);
			c.fill();

			c.fillStyle = "White";
			c.font = "12px Arial";
			c.fillText("55 mA", this.lx+40, this.y+5);	
		}
	}	
}

var fireflys = [];


function init(){
	for(var i=0; i < 8; i++){
		var xpos = 25+i%4*180;
		var ypos = 25+Math.floor(i/4)*350;
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
}

var selindex = -1;

function draw(selindex){
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
draw(selindex);

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


function update_boarddrawing(value) {
    var doffset = 6;
    var dnum = 2;

    motherboard.current = Number.parseFloat(value["crt1"][0]).toFixed(0) + "mA";
    motherboard.voltage = Number.parseFloat(value["crt1"][1]).toFixed(0) + "mV";

    motherboard.rxffvoltage = Number.parseFloat(value["crt1"][4]).toFixed(0) + "mV";
    motherboard.rxfftemp = Number.parseFloat(value["crt1"][3]).toFixed(0) + "C";

    for(var i=0; i < 12; i++){
        motherboard.rxlos[i] = value["crt1"][2] & (1<<i);
    }

    var dp = value["daughters present"];
    var fp = value["fireflys present"];
    for(var i=0; i < 8; i++){
        if((dp & (1<<i)) > 0){
            daughterboards[i].active =  true;
            for(var j=0; j < 3; j++){
                 if((fp[i] & (1<<j)) > 0){
                     fireflys[i*3+j].active = true;
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

function update_rxmask(value) {

    for(var i=0; i < 12; i++){
        motherboard.rxlosmask[i] = ((value & (1<<i)) != 0);
    }
    console.log(value + " " + motherboard.rxlosmask)
    draw(selindex);
}
