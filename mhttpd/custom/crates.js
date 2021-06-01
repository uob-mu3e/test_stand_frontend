var canvas = document.querySelector('canvas');
var cc = canvas.getContext('2d');



function Crate(x,y,dx,dy, index){
    this.x = x;
    this.y = y;
    this.dx= dx;
    this.dy= dy;

    this.index = index;


}

function FEB(x,y,dx,dy, index){
    this.x = x;
    this.y = y;
    this.dx= dx;
    this.dy= dy;

    this.index = index;


}


var crates = new Array(8);
var febs   = new Array(8);
for(var i=0; i < 8; i++){
    febs[i] = new Array(16);
    crates[i] = new Crate(i);
    for(var j=0; j < 16; j++){
        febs[i][j] = new Feb(10+xoffset*i, 40+yoffset*j, 280, 23, i*16+j);
    }
}
