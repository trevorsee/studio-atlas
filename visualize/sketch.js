/*
wiki-tSNE:: Wikipedia articles of political ideologies / philosophical concepts.
Converted to tf-idf matrix, then clustered by t-SNE.
*/

// first time you load the data, the exact positions have to be adjusted so set adjust to true
// after that, move the downloaded adjusted json file to the sketch folder and set adjust to be false
var filename = 'data_studios';

// parameters
var zoom = {x:1.5, y:3.0};
var margin = {box:8, l:80, t:50, r:350, b:70};
var tx = 120;
var ty = 85;
var infoMargin = 36;


////////////////////////////////////////
var canvas;
var data;
var boxes;
var info;
var tl, br;
var highlighted;
var infoHighlighted;
var ox, oy;
var x;
var y;

function preload() {
  data = loadJSON(filename+".json");
}

function setup() {
  var cw = 1440*1.5; //windowWidth;
  var ch = 800*1.5; //windowHeight;
  canvas = createCanvas(zoom.x * cw + margin.l + margin.r, zoom.y * ch + margin.t + margin.b);

  x = data.x;
  y = data.y;
  names = data.names;

  // draw the screen and turn off frame loop
  drawScreen();
  noLoop();
}


function drawScreen() {
  background(255);

  // draw boxes
  push();
  translate(-ox, -oy);

  for (var i=0; i<x.length; i++) {
    strokeWeight(1);
    fill(0);
    tempX = map(x[i],0,1,margin.l,width-margin.r)
    tempY = map(y[i],0,1,margin.t,height-margin.b)
    //rect(tempX, tempY, 10, 10);
    text(names[i],tempX, tempY);
  }
  pop();
}
