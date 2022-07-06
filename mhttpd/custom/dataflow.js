var canvas = document.querySelector('canvas');
var cc = canvas.getContext('2d');

var nlinksrx = [34,33,33,24];
var nlinkstx = [34,33,33,12];

function canvas_arrow(fromx, fromy, tox, toy, colour = "Black") {
    var headlen = 10; // length of head in pixels
    var dx = tox - fromx;
    var dy = toy - fromy;
    var angle = Math.atan2(dy, dx);
    cc.beginPath();
    cc.moveTo(fromx, fromy);
    cc.lineTo(tox, toy);
    cc.moveTo(tox, toy);
    cc.lineTo(tox - headlen * Math.cos(angle - Math.PI / 6), toy - headlen * Math.sin(angle - Math.PI / 6));
    cc.moveTo(tox, toy);
    cc.lineTo(tox - headlen * Math.cos(angle + Math.PI / 6), toy - headlen * Math.sin(angle + Math.PI / 6));
    cc.lineWidth = 1.0;
    cc.strokeStyle = colour;
    cc.stroke();
  }

  function RateMeter(x, y, counter, rate, ratemax, name = "", colour = "Blue"){
    this.x = x;
    this.y = y;
    this.w = 80;
    this.h = 8;
    this.counter = counter;
    this.lastCounter = 0;
    this.rate    = rate;
    this.ratemax = ratemax;
    this.nstrokes = ratemax;
    this.name = name;
    this.colour = colour;
    while(this.nstrokes > 10){
        this.nstrokes = this.nstrokes/4;
    }

    this.draw = function(colour = "Black"){
        if (colour != "Black")
            this.colour = "Gray";
        cc.fillStyle = this.colour;
        var fillSize = parseFloat(this.w)*this.rate/parseFloat(ratemax);
        if ( this.rate > this.ratemax )
            fillSize = parseFloat(this.w)*ratemax/parseFloat(ratemax);
        cc.fillRect(this.x, this.y, fillSize, this.h);

        cc.fillStyle = "Black";
        cc.font = "12px Arial, sans-serif";
        cc.textAlign = "center";
        cc.fillText(0, this.x, this.y-6);
        cc.fillText(this.ratemax, this.x+this.w, this.y-6);
        cc.textAlign = "left";
        cc.fillText(this.name, this.x, this.y+20);
        cc.textAlign = "right";
        cc.fillText(this.counter, this.x+this.w, this.y+20);

        
        for(var i=0; i <= this.nstrokes; i++){
            cc.beginPath();
            cc.moveTo(this.x + i*this.w/this.nstrokes, this.y);
            cc.lineTo(this.x + i*this.w/this.nstrokes, this.y-4);
            cc.lineWidth = 1;
            cc.strokeStyle = "Black";
            cc.stroke();
        }
        cc.textAlign = "left";
    }

    this.updateRate = function(counter, ts){
        var curDiff = Math.abs(this.lastCounter -  (new Uint32Array([counter]))[0]);
        this.rate = curDiff / ts;
        this.counter = curDiff;
        this.lastCounter = (new Uint32Array([counter]))[0];
    }
}


function Fifo(x,y, name, fullcounter =0, halffullcounter =0, hashalffull=true){
    this.x = x;
    this.y = y;
    this.w = 80;
    this.h = 20;
    this.fullcounter = fullcounter;
    this.halffullcounter    = halffullcounter;
    this.hashalffull = hashalffull;
    this.name = name;
    this.colour = "Green";

    this.draw = function(colour = "Black"){
        if(this.halffullcounter > 0)
            this.colour = "Orange";
        if(this.fullcounter > 0)
            this.colour = "Red";
        if (colour != "Black")
            this.colour = "Gray";
        cc.fillStyle = this.colour;
        cc.fillRect(this.x, this.y, this.w, this.h);

        cc.fillStyle = "Black";
        cc.font = "12px Arial, sans-serif";
        cc.textAlign = "left";
        cc.fillText(this.name, this.x+3, this.y+14);
        cc.textAlign = "right";
        cc.fillText(this.fullcounter, this.x+this.w, this.y+this.h+14);
        if(this.hashalffull)
            cc.fillText(this.halffullcounter, this.x+this.w, this.y+this.h+28);
        cc.textAlign = "left";
        cc.fillText("Full", this.x, this.y+this.h+14);
        if(this.hashalffull)
            cc.fillText("HF", this.x, this.y+this.h+28);       

    }
}

function FEBInput(x, y, name){
    this.x = x;
    this.y = y;
    this.colour = "Black";
    this.name = name;
    this.eventRate = new RateMeter(this.x+60, this.y+20, 0, 61036, 61036,"Evs");
    this.subheaderRate = new RateMeter(this.x+60, this.y+70, 0, 700000, 61036*128,"Subh");
    this.skippedRate = new RateMeter(this.x+200, this.y+70, 0, 6500, 61036, "Skip.", "Red");
    this.linkfifo = new Fifo(this.x+320, this.y+20, "Link FIFO")

    this.draw = function(){
        cc.fillStyle = this.colour;
        cc.font = "12px Arial, sans-serif";
        cc.fillText(this.name, this.x, this.y+50);
        this.eventRate.draw(this.colour);
        this.subheaderRate.draw(this.colour);
        this.skippedRate.draw(this.colour);
        this.linkfifo.draw(this.colour);
        if ( this.colour != "Black" ) {
            canvas_arrow(this.x+160, this.y+30, this.x+300, this.y+30, "Black");
            canvas_arrow(this.x+200, this.y+31, this.x+220, this.y+50, "Black");
        } else {
            canvas_arrow(this.x+160, this.y+30, this.x+300, this.y+30, "Blue");
            canvas_arrow(this.x+200, this.y+31, this.x+220, this.y+50, "Red");
        }
    }
}


function EventBuilder(x,y, name){
    this.x = x;
    this.y = y;
    this.name = name;

    this.idleNotHeaderCount = 0;
    this.bankBuilderTagFfifoFullCount =0;

    this.eventRate =  new RateMeter(this.x+200, this.y,0, 61036, 61036,"Evs");
    this.skippedEventRate = new RateMeter(this.x+200, this.y+40,0, 61036, 61036,"Skip.");

    this.draw = function(){
        cc.fillStyle = "Black";
        cc.font = "20px Arial, sans-serif";
        cc.fillText(this.name, this.x, this.y);

        cc.font = "12px Arial, sans-serif";
        cc.fillText("Idle count", this.x, this.y+25);
        cc.fillText("Tag FIFO full", this.x, this.y+45);    

        cc.textAlign = "right";
        cc.fillText(this.idleNotHeaderCount, this.x+160, this.y+25);
        cc.fillText(this.bankBuilderTagFfifoFullCount, this.x+160, this.y+45);   
        cc.textAlign = "left";

        this.eventRate.draw();
        this.skippedEventRate.draw();

    }
}

function RoundRobinRo(x,y, name){
    this.x = x;
    this.y = y;
    this.name = name;

    this.DebugFifo = new Fifo(this.x+100, this.y+20,"DebugFifo",0,0,false);
    this.EventBuilder = new EventBuilder(this.x+240, this.y+20, "Debug Event Builder");

    this.draw = function(){
        this.DebugFifo.draw();
        canvas_arrow(this.x+190, this.y+25, this.x+230, this.y+25,"Blue");
        this.EventBuilder.draw();

        cc.fillStyle = "Black";
        cc.font = "20px Arial, sans-serif";
        cc.fillText(this.name, this.x, this.y);

        cc.strokeStyle = "Black";
        cc.beginPath();
        cc.arc(this.x+40, this.y+60, 38, 2*Math.PI, false);
        cc.stroke();
    }
}

function TreeRo(x, y, name){
    this.x = x;
    this.y = y;
    this.name = name;

    this.layerFifos = [4, 2, 1];
    this.Outlayer = [];
    for ( var layer = 0; layer < this.layerFifos.length; layer++ ) {
        var curArray = []
        var curNumFifos = this.layerFifos[layer];
        for ( var fifo = 0; fifo < curNumFifos; fifo++ ) {
            curArray.push([
                (new Fifo(this.x+270*layer, this.y+20+150*fifo, "TreeL" + layer + "F" + fifo, 0, 0, false)),
                (new RateMeter(this.x+150+270*layer, this.y+20+150*fifo, 0, 61036, 61036,"Evs")),
                (new RateMeter(this.x+150+270*layer, this.y+70+150*fifo, 0, 700000, 61036*128,"Subh")),
                (new RateMeter(this.x+150+270*layer, this.y+120+150*fifo, 0, 700000, 61036*512,"Hits"))
            ])
        }
        this.Outlayer.push(curArray);
    }

    this.draw = function(){
        for ( var layer = 0; layer < this.layerFifos.length; layer++ ) {
            for ( var fifo = 0; fifo < this.Outlayer[layer].length; fifo++ ) {
                this.Outlayer[layer][fifo][0].draw(); // FIFO
                this.Outlayer[layer][fifo][1].draw(); // Evs
                this.Outlayer[layer][fifo][2].draw(); // Subh
                this.Outlayer[layer][fifo][3].draw(); // Hits
                canvas_arrow(this.x+90+270*layer, this.y+25+150*fifo, this.x+130+270*layer, this.y+25+150*fifo,"Blue");
            }
        }

        cc.fillStyle = "Black";
        cc.font = "20px Arial, sans-serif";
        cc.fillText(this.name, this.x, this.y);
    }
}

function GlobalTS(x,y){
    this.x = x;
    this.y = y;

    this.GlobalTSDiff = 0;
    this.LastTime = 0;

    this.draw = function(){
        cc.fillStyle = "Black";
        cc.font = "20px Arial, sans-serif";
        cc.fillText("GlobalTimeDiff", this.x, this.y);

        cc.font = "12px Arial, sans-serif";
        cc.fillText("GlobalTimeDiff", this.x, this.y+25);

        cc.textAlign = "right";
        cc.fillText(Math.round(this.GlobalTSDiff + "e+2") / 100 + "[s]", this.x+160, this.y+25); 
        cc.textAlign = "left";
    }
}

function DMA(x, y){
    this.x = x;
    this.y = y;

    this.DMAHalfFullCount = 0;

    this.draw = function(){
        cc.fillStyle = "Black";
        cc.font = "20px Arial, sans-serif";
        cc.fillText("DMA", this.x, this.y);

        cc.font = "12px Arial, sans-serif";
        cc.fillText("Half full count", this.x, this.y+25);

        cc.textAlign = "right";
        cc.fillText(this.DMAHalfFullCount, this.x+160, this.y+25); 
        cc.textAlign = "left";
    }
}

var febs = new Array(12);
for(var i=0; i <12; i++){
    febs[i] = new FEBInput(0,i*100,"FEB" + i);
}

var rrro = new RoundRobinRo(500, 50, "Round Robin RO Debug");

var tree = new Array(3);
tree[0] = new TreeRo(500, 200, "Tree Pixel US");
tree[1] = new TreeRo(500, 850, "Tree Pixel DS");
// tree[2] = new TreeRo(500, 650, "Tree Pixel Fibre");

var gts = new GlobalTS(1100, 50);

var dma = new DMA(1100, 100);

function init(){
    mjsonrpc_db_get_values(["/Equipment/SwicthingCentral/Variables/SCCN"]).then(function(rpc) {
        if(rpc.result.data[0]){
            update_sccn(rpc.result.data[0]);
        }  
     }).catch(function(error) {
        mjsonrpc_error_alert(error);
     });

     mjsonrpc_db_get_values(["/Equipment/LinksCentral/Settings/LinkMask"]).then(function(rpc) {
        if(rpc.result.data[0]){
            update_feb(rpc.result.data[0]);
        }  
     }).catch(function(error) {
        mjsonrpc_error_alert(error);
     });

}

function draw(){
    canvas = document.querySelector('canvas');
    cc = canvas.getContext('2d');

    cc.clearRect(0, 0, canvas.width, canvas.height);

    for(var i=0; i < 12; i++){
        febs[i].draw();
    }

    rrro.draw();

    tree[0].draw();
    tree[1].draw();

    dma.draw();

    gts.draw();
}

init();
draw();

function update_sccn(valuex){

    var value = valuex;
    if(typeof valuex === 'string')
        value = JSON.parse(valuex);

    // set global ts at the moment we only take the lower 32 bits, we save in sec
    gts.GlobalTSDiff = Math.abs((new Uint32Array([value[0]]))[0] - gts.LastTime) / 8 / 100000000; // (value[1] << 32) & 
    gts.LastTime = (new Uint32Array([value[0]]))[0];

    // TODO: add other detectors
    // first we do one detector
    rrro.DebugFifo.halffullcounter = 0; // SWB_STREAM_DEBUG_FIFO_ALFULL_CNT --> we dont have this for now
    rrro.EventBuilder.idleNotHeaderCount = value[2]; // SWB_BANK_BUILDER_IDLE_NOT_HEADER_CNT
    rrro.EventBuilder.skippedEventRate.updateRate(value[3], gts.GlobalTSDiff); // SWB_BANK_BUILDER_SKIP_EVENT_CNT
    rrro.EventBuilder.eventRate.updateRate(value[4], gts.GlobalTSDiff); // SWB_BANK_BUILDER_EVENT_CNT
    rrro.EventBuilder.bankBuilderTagFfifoFullCount = value[5]; // SWB_BANK_BUILDER_TAG_FIFO_FULL_CNT

    // TODO: add SWB_MERGER_DEBUG_FIFO_ALFULL_CNT

    var offeset = 6;
    for ( var d = 0; d < 2; d++ ) {
        // skip the stream fifo
        offeset += 8;
        // setup input link fifos
        for(var i=0; i < 5; i++){
            offeset += 1;
            febs[i+d*5].linkfifo.halffullcounter = value[offeset + 0]; // SWB_LINK_FIFO_ALMOST_FULL_CNT
            febs[i+d*5].linkfifo.fullcounter = value[offeset + 1]; // SWB_LINK_FIFO_FULL_CNT
            febs[i+d*5].skippedRate.updateRate(value[offeset + 2], gts.GlobalTSDiff); // SWB_SKIP_EVENT_CNT
            febs[i+d*5].eventRate.updateRate(value[offeset + 3], gts.GlobalTSDiff); // SWB_EVENT_CNT
            febs[i+d*5].subheaderRate.updateRate(value[offeset + 4], gts.GlobalTSDiff); // SWB_SUB_HEADER_CNT
            // TODO: add ReadBackMergerRate
            offeset += 3+5;
        }

        // setup tree counters
        for ( var layer = 0; layer < tree[0].layerFifos.length; layer++ ) {
            offeset += 1;
            for ( var fifo = 0; fifo < tree[0].Outlayer[layer].length; fifo++ ) {
                offeset += 1;
                // TODO: add fifo flags tree[0].Outlayer[layer][fifo][0]
                console.log("L", layer, "F", fifo, value[offeset + 0], offeset);
                tree[d].Outlayer[layer][fifo][1].updateRate(value[offeset + 0], gts.GlobalTSDiff); // Evs
                tree[d].Outlayer[layer][fifo][2].updateRate(value[offeset + 1], gts.GlobalTSDiff); // Subh
                tree[d].Outlayer[layer][fifo][3].updateRate(value[offeset + 2], gts.GlobalTSDiff); // Hits
                offeset += 3;
            }
        }
    }

    draw();
}


function update_feb(valuex){
    var value = valuex;
    if(typeof valuex === 'string')
        value = JSON.parse(valuex);

    for(var i=0; i < 12; i++){
        if (value[i] != 3)
            febs[i].colour = "Gray";
    }
}
