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

  function RateMeter(x,y, counter, rate, ratemax, name = "", colour = "Blue"){
    this.x = x;
    this.y = y;
    this.w = 80;
    this.h = 8;
    this.counter = counter;
    this.rate    = rate;
    this.ratemax = ratemax;
    this.nstrokes = ratemax;
    this.name = name;
    this.colour = colour;
    while(this.nstrokes > 10){
        this.nstrokes = this.nstrokes/4;
    }

    this.draw = function(){
        cc.fillStyle = this.colour;
        cc.fillRect(this.x, this.y, this.w*rate/ratemax, this.h);

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

    this.draw = function(){
        if(this.halffullcounter > 0)
            this.colour = "Orange";
        if(this.fullcounter > 0)
            this.colour = "Red";
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
    this.name = name;
    this.eventRate = new RateMeter(this.x+60, this.y+20,0, 61036, 61036,"Evs");
    this.subheaderRate = new RateMeter(this.x+60, this.y+70,0,700000, 61036*128,"Subh");
    this.skippedRate = new RateMeter(this.x+200, this.y+70,0, 6500, 61036, "Skipped", "Red");
    this.linkfifo = new Fifo(this.x+320, this.y+20, "Link FIFO")

    this.draw = function(){
        cc.fillStyle = "Black";
        cc.font = "12px Arial, sans-serif";
        cc.fillText(this.name, this.x, this.y+50);
        this.eventRate.draw();
        this.subheaderRate.draw();
        this.skippedRate.draw();
        this.linkfifo.draw();
        canvas_arrow(this.x+160, this.y+30, this.x+300, this.y+30,"Blue");
        canvas_arrow(this.x+200, this.y+31, this.x+220, this.y+50,"Red");
    }
}


function EventBuilder(x,y, name){
    this.x = x;
    this.y = y;
    this.name = name;

    this.idleNotHeaderCount = 0;
    this.bankBuilderTagFfifoFullCount =0;

    this.eventRate =  new RateMeter(this.x+200, this.y,0, 61036, 61036,"Evs");
    this.skippedEventRate = new RateMeter(this.x+200, this.y+40,0, 61036, 61036,"Skipped");

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

    this.FarmFifo = new Fifo(this.x+100, this.y+20,"FarmFifo",0,0,false);
    this.DebugFifo = new Fifo(this.x+100, this.y+60,"DebugFifo",0,0,false);
    this.FarmEventRate = new RateMeter(this.x+240, this.y+20,0, 61036, 61036,"Evs");

    this.EventBuilder = new EventBuilder(this.x+240, this.y+80, "Debug Event Builder");

    this.draw = function(){
        this.FarmFifo.draw();
        this.DebugFifo.draw();
        canvas_arrow(this.x+190, this.y+25, this.x+230, this.y+25,"Blue");
        this.FarmEventRate.draw();
        canvas_arrow(this.x+190, this.y+70, this.x+230, this.y+70,"Blue");
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

function DMA(x,y){
    this.x = x;
    this.y = y;

    this.DMAHalfFullCount =0;

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

var rrro = new Array(3);
rrro[0] = new RoundRobinRo(500,50, "Round Robin RO Pixel US");
rrro[1] = new RoundRobinRo(500,250, "Round Robin RO Pixel DS");
rrro[2] = new RoundRobinRo(500,450, "Round Robin RO Fibre");

var dma = new DMA(1100,50);

function init(){

}

function draw(){
    for(var i=0; i <12; i++){
        febs[i].draw();
    }
    for(var i=0; i <3; i++){
        rrro[i].draw();
    }

    dma.draw();    
}

init();
draw();



