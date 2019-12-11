var canvas = document.querySelector('canvas');

var c = canvas.getContext('2d');

function RXLink(xmin,y,xmax, index, name){
    this.status =0;
    this.type="Undefined type";
    this.shorttype="U";
    this.selected = false;
    this.xmin = xmin;
    this.y = y;
    this.ymin = y-5;
    this.xmax = xmax;
    this.ymax = y+5;

    this.lxmin = xmin-10;
    this.lymin = y-10;
    this.lxmax = xmax+60;
    this.lymax = y+90;

    this.bxmin = this.lxmin+10;
    this.bymin = this.lymin+60;
    this.bxmax = this.lxmin+70;
    this.bymax = this.lymin+90;

    this.col = "rgb(150,150,150)"
    this.name = name;
    this.index = index;

    this.draw = function(){
        if(this.status == 0)
            this.col = "rgb(150,150,150)";
        if(this.status == 1)
            this.col = "rgb(100,200,100)";
        if(this.status == 2)
            this.col = "rgb(255,165,0)";
        if(this.status == 3)
            this.col = "rgb(255,0,0)";

        if(this.selected == false){
            c.beginPath();
            c.moveTo(this.xmin, this.y);
            c.lineTo(this.xmax, this.y);
            c.lineWidth = 6;
            c.strokeStyle = this.col;
            c.stroke();

            c.fillStyle = "Black";
            if(this.shorttype == "P")
               c.fillStyle = "Blue";
            if(this.shorttype == "F")
               c.fillStyle = "Magenta";
            if(this.shorttype == "T")
               c.fillStyle = "Cyan";
            c.font = "8px Arial";
            c.fillText(this.shorttype, this.xmin-8, this.y+2);
        } else {
            c.fillStyle = this.col;
            c.fillRect(this.lxmin, this.lymin,this.lxmax-this.lxmin, this.lymax-this.lymin);
            c.fillStyle = "Black";
            c.font = "12px Arial";
            c.fillText(this.name, this.lxmin+10, this.lymin+15);
            c.fillStyle = "Black";
            c.font = "12px Arial";
            c.fillText(this.type, this.lxmin+10, this.lymin+30);


            c.fillStyle = "rgb(50,50,50)";
            c.fillRect(this.bxmin, this.bymin,this.bxmax-this.bxmin, this.bymax-this.bymin);
            c.fillStyle = "White";
            c.font = "12px Arial";
            if(this.status == 0){
                c.fillText("Enable", this.bxmin+10, this.bymin+17);
            } else {
                c.fillText("Disable", this.bxmin+10, this.bymin+17);
            }

        }

    }
}

var rxlinks = new Array(192);

function Switchingboard(x,y,dx,dy, index){
	this.x = x;
	this.y = y;
    this.dx= dx;
	this.dy= dy;

    this.bxmin = this.x+30;
    this.bymin = this.y+760;
    this.bxmax = this.x+100;
    this.bymax = this.y+790;

    this.index = index;
    this.active = 0;
    this.name = "";

    this.RX = new Array(4);
    this.TX = new Array(4);
    for(var i =0; i < 4; i++){
       this.RX[i] = new RXPod(this.x+10, this.y+10+170*i,40, 160,i, index);
       this.TX[i] = new TXPod(this.x+65, this.y+10+170*i,40, 160,i, index);
    }


	this.draw = function(){
        c.fillStyle = "rgba(0,0,0,0.5)";
        c.fillRect(this.x+10, this.y+10,this.dx, this.dy);
        if(this.active > 0)
            c.fillStyle = "rgb(80,80,180)";
        else
            c.fillStyle = "rgb(200,200,200)";
		c.fillRect(this.x, this.y,this.dx, this.dy);

        c.fillStyle = "Black";
        c.font = "20px Arial";
        c.fillText(this.index, this.x+10, this.y+780);
        if(this.active > 0){
            for(var i =0; i < 4; i++){
                this.RX[i].draw();
                this.TX[i].draw();
            }
        }


        c.fillStyle = "Black";
        c.font = "16px Arial";
        c.fillText("RX", this.x+20, this.y+700);

        c.fillStyle = "Black";
        c.font = "16px Arial";
        c.fillText("TX", this.x+75, this.y+700);

        c.fillStyle = "Black";
        c.font = "12px Arial";
        c.fillText(this.name, this.x+10, this.y+740);

        c.fillStyle = "rgb(50,50,50)";
        c.fillRect(this.bxmin, this.bymin,this.bxmax-this.bxmin, this.bymax-this.bymin);
        c.fillStyle = "White";
        c.font = "12px Arial";
        if(this.active == 0){
            c.fillText("Enable", this.bxmin+12, this.bymin+19);
        } else {
            c.fillText("Disable", this.bxmin+12, this.bymin+19);
        }

	}
}



function RXPod(x,y,dx,dy, name, swboard){
    this.x = x;
    this.y = y;
    this.dx= dx;
    this.dy= dy;
    this.name = name;
    this.links = [12];



    for(var i=0; i < 12; i ++){
        rxlinks[swboard*48+name*12+i] = new RXLink(this.x-40, this.y+10+i*12, this.x+10, swboard*48+name*12+i,"FEB");
        this.links[i] = rxlinks[swboard*48+name*12+i];
    }


    this.draw = function(){
        c.fillStyle = "rgb(80,200,120)";
        c.fillRect(this.x, this.y,this.dx, this.dy);

        for(var i=0; i < 12; i ++){
           this.links[i].draw();
        }
    }
}



function TXPod(x,y,dx,dy, name){
    this.x = x;
    this.y = y;
    this.dx= dx;
    this.dy= dy;
    this.name = name;
Math.floor(y/x);
    this.draw = function(){
        c.fillStyle = "rgb(120,200,80)";
        c.fillRect(this.x, this.y,this.dx, this.dy);
    }
}

var switchingboards = new Array(4);
switchingboards[0] = new Switchingboard( 50,25,120,800,0);
switchingboards[1] = new Switchingboard(350,25,120,800,1);
switchingboards[2] = new Switchingboard(650,25,120,800,2);
switchingboards[3] = new Switchingboard(950,25,120,800,3);

function init(){

    mjsonrpc_db_get_values(["/Equipment/Links/Settings"]).then(function(rpc) {
       update_masks(rpc.result.data[0]);
    }).catch(function(error) {
       mjsonrpc_error_alert(error);
    });
    draw(-1);
}

var rxselindex = -1;

function draw(rxselindex){
	c.fillStyle = "Silver";
	c.fillRect(0, 0, canvas.width, canvas.height);

    for(var i=0; i < 4; i++){
        switchingboards[i].draw();
    }

    if(rxselindex > -1)
        rxlinks[rxselindex].draw();
}

init();
draw(rxselindex);

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
        if(mouse.x >= switchingboards[i].bxmin &&
            mouse.x <= switchingboards[i].bxmax &&
            mouse.y >= switchingboards[i].bymin &&
            mouse.y <= switchingboards[i].bymax ){
            if(switchingboards[i].active==0){
                switchingboards[i].active=1;
            } else {
                switchingboards[i].active=0;
            }
            mjsonrpc_db_set_value("/Equipment/Links/Settings/SwitchingBoardMask["+i+"]", switchingboards[i].active);
            found = true;
        }

    }

    for(var i=1000*found; i < 192; i++){
        if(rxlinks[i].selected && !found){
             if(rxselindex == i){
                    if(mouse.x >= rxlinks[i].lxmin &&
                        mouse.x <= rxlinks[i].lxmax &&
                        mouse.y >= rxlinks[i].lymin &&
                        mouse.y <= rxlinks[i].lymax ){
                            // check for button
                            if(mouse.x >= rxlinks[i].bxmin &&
                                mouse.x <= rxlinks[i].bxmax &&
                                mouse.y >= rxlinks[i].bymin &&
                                mouse.y <= rxlinks[i].bymax ){
                                if(rxlinks[i].status==0){
                                    rxlinks[i].status=1;
                                } else {
                                    rxlinks[i].status=0;
                                }
                                mjsonrpc_db_set_value("/Equipment/Links/Settings/FrontEndBoardMask["+i+"]", rxlinks[i].status);
                                found = true;

                            } else {
                                rxselindex = -1;
                                found = true;
                                rxlinks[i].selected = false;
                            }
                 }
             }
            } else {
                if( found == false &&
                        mouse.x >= rxlinks[i].xmin &&
                       mouse.x <= rxlinks[i].xmax &&
                       mouse.y >= rxlinks[i].ymin &&
                       mouse.y <= rxlinks[i].ymax){
                       if(rxselindex > -1)
                           rxlinks[rxselindex].selected = false;
                       rxselindex = i;
                       found = true;
                       rxlinks[i].selected = true;
                    }
            }

        }
    console.log(mouse,found,rxselindex);
    if(found)
        draw(rxselindex);
})


function update_boarddrawing(value) {
    var swmask = value["switchingboardstatus"];
    for(var i=0; i < 4; i++){
        switchingboards[i].active = swmask[i];
    }

    var femask = value["frontendboardstatus"];
    for(var i=0; i < 192; i++){
       rxlinks[i].status = femask[i];
    }

    draw(rxselindex);
}

function update_masks(value) {
    var swmask = value["switchingboardmask"];
    var swnames = value["switchingboardnames"];
    for(var i=0; i < 4; i++){
        switchingboards[i].active = swmask[i];
        switchingboards[i].name  = swnames[i];
    }
    var femask = value["frontendboardmask"];
    var fenames = value["frontendboardnames"];
    var fetypes = value["frontendboardtype"];
    for(var i=0; i < 192; i++){
        if(femask[i] == 0)
           rxlinks[i].status = 0;
        else {
            if(rxlinks[i].status == 0)
                rxlinks[i].status = 2;
        }
        rxlinks[i].name = fenames[i];
        if(fetypes[i] == 1){
            rxlinks[i].type = "Pixel";
            rxlinks[i].shorttype = "P";
        } else if   (fetypes[i] == 2){
            rxlinks[i].type = "Fibre";
            rxlinks[i].shorttype = "F";
        } else if   (fetypes[i] == 3){
            rxlinks[i].type = "Tiles";
            rxlinks[i].shorttype = "T";
        } else {
            rxlinks[i].type = "Undefined Type";
            rxlinks[i].shorttype = "U";
        }
    }
    draw(rxselindex);
}
