var canvas = document.querySelector('canvas');
var cc = canvas.getContext('2d');

function DataLink(xmin,y,xmax, index, name){
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
            cc.beginPath();
            cc.moveTo(this.xmin, this.y);
            cc.lineTo(this.xmax, this.y);
            cc.lineWidth = 6;
            cc.strokeStyle = this.col;
            cc.stroke();

            cc.fillStyle = "Black";
            if(this.shorttype == "P")
               cc.fillStyle = "Blue";
            if(this.shorttype == "F")
               cc.fillStyle = "Magenta";
            if(this.shorttype == "T")
               cc.fillStyle = "Cyan";
            if(this.shorttype == "FS")
               cc.fillStyle = "Magenta";
            cc.font = "8px Arial";
            cc.fillText(this.shorttype, this.xmin-8, this.y+2);
        } else {
            cc.fillStyle = this.col;
            cc.fillRect(this.lxmin, this.lymin,this.lxmax-this.lxmin, this.lymax-this.lymin);
            cc.fillStyle = "Black";
            cc.font = "12px Arial, sans-serif";
            cc.fillText(this.name, this.lxmin+10, this.lymin+15);
            cc.fillStyle = "Black";
            cc.font = "12px Arial";
            cc.fillText(this.type, this.lxmin+10, this.lymin+30);


            cc.fillStyle = "rgb(50,50,50)";
            cc.fillRect(this.bxmin, this.bymin,this.bxmax-this.bxmin, this.bymax-this.bymin);
            cc.fillStyle = "White";
            cc.font = "12px Arial, sans-serif";
            if(this.status == 0){
                cc.fillText("Enable", this.bxmin+10, this.bymin+17);
            } else {
                cc.fillText("Disable", this.bxmin+10, this.bymin+17);
            }

        }

    }
}


function SCLink(xmin,y,xmax, index, name){
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
            cc.beginPath();
            cc.moveTo(this.xmin, this.y);
            cc.lineTo(this.xmax, this.y);
            cc.lineWidth = 6;
            cc.strokeStyle = this.col;
            cc.stroke();

            cc.fillStyle = "Black";
            if(this.shorttype == "P")
               cc.fillStyle = "Blue";
            if(this.shorttype == "F")
               cc.fillStyle = "Magenta";
            if(this.shorttype == "T")
               cc.fillStyle = "Cyan";
            if(this.shorttype == "FS")
               cc.fillStyle = "Magenta";
            cc.font = "8px Arial";
            cc.fillText(this.shorttype, this.xmax+8, this.y+2);
        } else {
            cc.fillStyle = this.col;
            cc.fillRect(this.lxmin, this.lymin,this.lxmax-this.lxmin, this.lymax-this.lymin);
            cc.fillStyle = "Black";
            cc.font = "12px Arial, sans-serif";
            cc.fillText(this.name, this.lxmin+10, this.lymin+15);
            cc.fillStyle = "Black";
            cc.font = "12px Arial, sans-serif";
            cc.fillText(this.type, this.lxmin+10, this.lymin+30);

            if(rxlinks[index].shorttype != "FS"){
                cc.fillStyle = "rgb(50,50,50)";
                cc.fillRect(this.bxmin, this.bymin,this.bxmax-this.bxmin, this.bymax-this.bymin);
                cc.fillStyle = "White";
                cc.font = "12px Arial, sans-serif";
                if(this.status == 0){
                    cc.fillText("Enable", this.bxmin+10, this.bymin+17);
                } else {
                    cc.fillText("Disable", this.bxmin+10, this.bymin+17);
                }
            }

        }

    }
}


var rxlinks = new Array(192);
var txlinks = new Array(192);

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

    this.Data = new Array(4);
    this.SC = new Array(4);
    for(var i =0; i < 4; i++){
       this.Data[i] = new DataPod(this.x+10, this.y+10+170*i,40, 160,i, index);
       this.SC[i] = new SCPod(this.x+65, this.y+10+170*i,40, 160,i, index);
    }


	this.draw = function(){
        cc.fillStyle = "rgba(0,0,0,0.5)";
        cc.fillRect(this.x+10, this.y+10,this.dx, this.dy);
        if(this.active > 0)
            cc.fillStyle = "rgb(80,80,180)";
        else
            cc.fillStyle = "rgb(200,200,200)";
        cc.fillRect(this.x, this.y,this.dx, this.dy);

        cc.fillStyle = "Black";
        cc.font = "20px Arial, sans-serif";
        cc.fillText(this.index, this.x+10, this.y+780);
        if(this.active > 0){
            for(var i =0; i < 4; i++){
                this.Data[i].draw();
                this.SC[i].draw();
            }
        }


        cc.fillStyle = "Black";
        cc.font = "16px Arial, sans-serif";
        cc.fillText("Data", this.x+20, this.y+700);

        cc.fillStyle = "Black";
        cc.font = "16px Arial, sans-serif";
        cc.fillText("SC", this.x+75, this.y+700);

        cc.fillStyle = "Black";
        cc.font = "12px Arial, sans-serif";
        cc.fillText(this.name, this.x+10, this.y+740);

        cc.fillStyle = "rgb(50,50,50)";
        cc.fillRect(this.bxmin, this.bymin,this.bxmax-this.bxmin, this.bymax-this.bymin);
        cc.fillStyle = "White";
        cc.font = "12px Arial, sans-serif";
        if(this.active == 0){
            cc.fillText("Enable", this.bxmin+12, this.bymin+19);
        } else {
            cc.fillText("Disable", this.bxmin+12, this.bymin+19);
        }

	}
}



function DataPod(x,y,dx,dy, name, swboard){
    this.x = x;
    this.y = y;
    this.dx= dx;
    this.dy= dy;
    this.name = name;
    this.links = [12];



    for(var i=0; i < 12; i ++){
        rxlinks[swboard*48+name*12+i] = new DataLink(this.x-40, this.y+10+i*12, this.x+10, swboard*48+name*12+i,"FEB");
        this.links[i] = rxlinks[swboard*48+name*12+i];
    }


    this.draw = function(){
        cc.fillStyle = "rgb(80,200,120)";
        cc.fillRect(this.x, this.y,this.dx, this.dy);

        for(var i=0; i < 12; i ++){
           this.links[i].draw();
        }
    }
}



function SCPod(x,y,dx,dy, name, swboard){
    this.x = x;
    this.y = y;
    this.dx= dx;
    this.dy= dy;
    this.name = name;
    this.links = [12];

    for(var i=0; i < 12; i ++){
        txlinks[swboard*48+name*12+i] = new SCLink(this.x+this.dx-10, this.y+10+i*12, this.x+this.dx+40, swboard*48+name*12+i,"FEB");
        this.links[i] = txlinks[swboard*48+name*12+i];
    }



    this.draw = function(){
        cc.fillStyle = "rgb(120,200,80)";
        cc.fillRect(this.x, this.y,this.dx, this.dy);

        for(var i=0; i < 12; i ++){
           this.links[i].draw();
        }

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
var txselindex = -1;

function draw(rxselindex, txselindex){
    canvas = document.querySelector('canvas');
    cc = canvas.getContext('2d');

    cc.fillStyle = "Silver";
    cc.fillRect(0, 0, canvas.width, canvas.height);

    for(var i=0; i < 4; i++){
        switchingboards[i].draw();
    }

    if(rxselindex > -1)
        rxlinks[rxselindex].draw();
    if(txselindex > -1)
        txlinks[txselindex].draw();

}

init();
draw(rxselindex, txselindex);

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
                                mjsonrpc_db_set_value("/Equipment/Links/Settings/LinkMask["+i+"]", 3*rxlinks[i].status);
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

        if(txlinks[i].selected && !found){
             if(txselindex == i){
                    if(mouse.x >= txlinks[i].lxmin &&
                        mouse.x <= txlinks[i].lxmax &&
                        mouse.y >= txlinks[i].lymin &&
                        mouse.y <= txlinks[i].lymax ){
                            // check for button
                            if(mouse.x >= txlinks[i].bxmin &&
                                mouse.x <= txlinks[i].bxmax &&
                                mouse.y >= txlinks[i].bymin &&
                                mouse.y <= txlinks[i].bymax ){
                                if(txlinks[i].status==0){
                                    txlinks[i].status=1;
                                } else {
                                    txlinks[i].status=0;
                                }
                                mjsonrpc_db_set_value("/Equipment/Links/Settings/LinkMask["+i+"]", txlinks[i].status);
                                found = true;

                            } else {
                                txselindex = -1;
                                found = true;
                                txlinks[i].selected = false;
                            }
                 }
             }
            } else {
                if( found == false &&
                        mouse.x >= txlinks[i].xmin &&
                       mouse.x <= txlinks[i].xmax &&
                       mouse.y >= txlinks[i].ymin &&
                       mouse.y <= txlinks[i].ymax){
                       if(txselindex > -1)
                           txlinks[txselindex].selected = false;
                       txselindex = i;
                       found = true;
                       txlinks[i].selected = true;
                    }
            }


        }
    if(found)
        draw(rxselindex, txselindex);
})


function update_boarddrawing(valuex) {

    var value = valuex;
    if(typeof valuex === 'string')
        value = JSON.parse(valuex);

    var swmask = value["switchingboardstatus"];
    for(var i=0; i < 4; i++){
        switchingboards[i].active = swmask[i];
    }

    var rxstat = value["rxlinkstatus"];
    var txstat = value["txlinkstatus"];
    for(var i=0; i < 192; i++){
       rxlinks[i].status = rxstat[i];
       txlinks[i].status = txstat[i];
    }

    draw(rxselindex, txselindex);
}

function update_masks(valuex) {



    var value = valuex;
    if(typeof valuex === 'string')
        value = JSON.parse(valuex);


    var swmask = value["switchingboardmask"];
    var swnames = value["switchingboardnames"];
    for(var i=0; i < 4; i++){
        switchingboards[i].active = swmask[i];
        switchingboards[i].name  = swnames[i];
    }
    var femask = value["linkmask"];
    var fenames = value["frontendboardnames"];
    var fetypes = value["frontendboardtype"];
    for(var i=0; i < 192; i++){
        if(femask[i] == 0){
           rxlinks[i].status = 0;
           txlinks[i].status = 0;
        } else if(femask[i] == 1){
            if(txlinks[i].status == 0)
                txlinks[i].status = 2;
            rxlinks[i].status = 0;
        } else {
            if(rxlinks[i].status == 0)
                rxlinks[i].status = 2;
            if(txlinks[i].status == 0)
                txlinks[i].status = 2;
        }
        rxlinks[i].name = fenames[i];
        txlinks[i].name = fenames[i];
        if(fetypes[i] == 1){
            rxlinks[i].type = "Pixel";
            rxlinks[i].shorttype = "P";
            txlinks[i].type = "Pixel";
            txlinks[i].shorttype = "P";

        } else if   (fetypes[i] == 2){
            rxlinks[i].type = "Fibre";
            rxlinks[i].shorttype = "F";
            txlinks[i].type = "Fibre";
            txlinks[i].shorttype = "F";

        } else if   (fetypes[i] == 3){
            rxlinks[i].type = "Tiles";
            rxlinks[i].shorttype = "T";
            txlinks[i].type = "Tiles";
            txlinks[i].shorttype = "T";

        } else if   (fetypes[i] == 4){
            rxlinks[i].type = "Fibre 2nd";
            rxlinks[i].shorttype = "FS";
            txlinks[i].type = "Unconnected";
            txlinks[i].shorttype = "U";
            txlinks[i].status = 0;
        } else {
            rxlinks[i].type = "Undefined Type";
            rxlinks[i].shorttype = "U";
        }
    }
    draw(rxselindex, txselindex);
}
