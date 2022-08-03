var canvas = document.querySelector('canvas');
var cc = canvas.getContext('2d');

function arrival_histogram(x,y){
    this.x = x;
    this.y = y;
    this.w = 16;
    this.h = 25;

    this.data = new Array(4);
    this.data[0]=0;
    this.data[1]=0;
    this.data[2]=0;
    this.data[3]=0;

    this.draw = function(){
        var max =0;
        var maxbin = -1;
        for(var i =0; i < 4; i++){
            if(this.data[i] >= max){
                max = this.data[i];
                maxbin = i;
            }
        }

        for(var i =0; i < 4; i++){
            if(i==maxbin)
                cc.fillStyle = "Green";
            else
                cc.fillStyle = "Red";
            var drawval = this.h*this.data[i]/max;
            if(this.data[i] > 0 && drawval < 1)
                drawval =1;
            cc.fillRect(this.x+this.w/4*i, this.y+this.h, this.w/4, -drawval);
        }
    }
}

function lvdslink(x,y,name){
    this.x = x;
    this.y = y;
    this.name = name;
    this.locked = 0;
    this.ready  = 0;

    this.disperr =0;
    this.disperr_last =0;

    this.numhits =0;
    this.numhits_last =0;

    this.ah = new arrival_histogram(this.x, this.y);

    this.draw = function(){
        this.ah.draw();

        cc.fillStyle = "Black";
        cc.font = "12px Arial, sans-serif";
        cc.textAlign = "left";
        cc.fillText(this.name, this.x+20, this.y+10);

        cc.font = "10px Arial, sans-serif";
        cc.textAlign = "left";
        cc.fillText("L", this.x+20, this.y+24);

        cc.fillStyle = this.locked ? "Green" : "Red";
        cc.beginPath();
        cc.arc(this.x+33, this.y+20, 5, 2*Math.PI, false);
        cc.fill();

        cc.fillStyle = "Black";
        cc.textAlign = "left";
        cc.fillText("R", this.x+45, this.y+24);

        cc.fillStyle = this.ready ? "Green" : "Red";
        cc.beginPath();
        cc.arc(this.x+58, this.y+20, 5, 2*Math.PI, false);
        cc.fill();       

        cc.fillStyle = (this.disperr-this.disperr_last > 0) ? "Red" : "Green";
        cc.font = "12px Arial, sans-serif";
        cc.textAlign = "right";
        cc.fillText(this.disperr-this.disperr_last, this.x+120, this.y+10);
        this.disperr_last = this.disperr;

        cc.fillStyle = "Black";
        cc.fillText(this.numhits-this.numhits_last, this.x+120, this.y+28);
        this.numhits_last = this.numhits;
    }
}

function feb(x,y,name){
    this.x = x;
    this.y = y;
    this.name = name;

    this.links = new Array(36);

    for(var i=0; i<36; i++){
        this.links[i] = new lvdslink(this.x,this.y+30+i*32, "Link "+ i);
    }


    this.draw = function(){

        cc.fillStyle = "Black";
        cc.font = "20px Arial, sans-serif";
        cc.textAlign = "left";
        cc.fillText(this.name, this.x, this.y+20);


        for(var i=0; i<36; i++){
            this.links[i].draw();
        }

    }
}

var febs = Array(10);
for(var i=0; i < 10; i++)
    febs[i] = new feb(160*i,0,"FEB "+ i);

function init(){
    mjsonrpc_db_get_values(["/Equipment/PixelsCentral/Variables/PCLS"]).then(function(rpc) {
        if(rpc.result.data[0]){
            update_pcls(rpc.result.data[0]);
        }  
     }).catch(function(error) {
        mjsonrpc_error_alert(error);
     });

}

function draw(){
    canvas = document.querySelector('canvas');
    cc = canvas.getContext('2d');

    cc.fillStyle = "#f2f2f2";
    cc.fillRect(0, 0, canvas.width, canvas.height);

    for(var i=0; i < 10; i++)
        febs[i].draw();

        var c = document.getElementById("myCanvas");
        var ctx = c.getContext("2d");
        ctx.moveTo(0, 0);
        ctx.lineTo(200, 100);
        ctx.stroke();

}


init();
draw();

function update_pcls(valuex){
    var value = valuex;
    if(typeof valuex === 'string')
        value = JSON.parse(valuex);

    var offset = 0;
    var nlinks = 36;
    for(var f=0; f <10; f++){
        // get the FEB.GetLinkID() from the bank and only update this FEB
        var febindex = parseInt(value[offset+1],16);
        if(f != febindex)
            continue;
        for(var l=0; l < nlinks; l++){
            console.log(f + " " + l);
            febs[f].links[l].locked = value[offset+2+6*l] & (1<<28);
            febs[f].links[l].ready  = value[offset+2+6*l] & (1<<31);
            febs[f].links[l].disperr= value[offset+2+6*l] & 0xFFFFFFF;
            febs[f].links[l].numhits= value[offset+3+6*l];
            febs[f].links[l].ah.data[0] = value[offset+4+6*l];
            febs[f].links[l].ah.data[1] = value[offset+5+6*l];
            febs[f].links[l].ah.data[2] = value[offset+6+6*l];
            febs[f].links[l].ah.data[3] = value[offset+7+6*l];
        }
        offset += 2 + nlinks*6;
        if(offset >= value.length)
            break;
    }
    draw();
}



