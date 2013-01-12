import oscP5.*;
import netP5.*;
import dmxP512.*;
import processing.serial.*;

DmxP512 dmxOutput;
OscP5 oscP5;
PImage myImage;
int universeSize = 128;
int myListeningPort = 32000;


color m_red = color(128, 0, 0);
color m_green = color(0, 128, 0);
color m_blue = color(0, 0, 128);

HashMap hashmapTime = new HashMap();
HashMap hashmapDirection = new HashMap();

String DMXPRO_PORT = "COM14";
int DMXPRO_BAUDRATE = 115000;

void setup() {
  myImage = loadImage("tweetingcolours-640x254.png");
  size(myImage.width, myImage.height);
  image(myImage, 0, 0);

  //drawScene();
  oscP5 = new OscP5(this, myListeningPort);
  NetInfo.wan();
  
//  dmxOutput = new DmxP512(this, universeSize, false);
//  dmxOutput.setupDmxPro(DMXPRO_PORT,DMXPRO_BAUDRATE);
//  
//  dmxOutput.set(1, 200);
//  dmxOutput.set(3, 200);
//  
//  dmxOutput.set(5, 0);
//  dmxOutput.set(6, 0);
//  dmxOutput.set(7, 0);     
//  dmxOutput.set(8, 0);     
//   
//  dmxOutput.set(10, 0);
//  dmxOutput.set(11, 0);
//  dmxOutput.set(12, 0);    
//  dmxOutput.set(13, 0); 

  frameRate(25);
}

void draw() {
  image(myImage, 0, 0);
  Iterator i = hashmapDirection.entrySet().iterator();
  

//////////////// Mouse gaze
/*
      int mx = mouseX;
      int my = mouseY;
      color ct = get(mx, my);
      drawCross(mx, my, ct);
      
      System.out.println("r, g, b : " + (int)red(ct) + ", " + (int)green(ct) + ", " + (int)blue(ct));
      
      dmxOutput.set(5, (int)red(ct));
      dmxOutput.set(6, (int)green(ct));
      dmxOutput.set(7, (int)blue(ct));     
     
      dmxOutput.set(10, (int)red(ct));
      dmxOutput.set(11, (int)green(ct));
      dmxOutput.set(12, (int)blue(ct));
*/
///////////////////

  while (i.hasNext()) {
    Map.Entry me = (Map.Entry)i.next();
    if (System.currentTimeMillis() - (Long)hashmapTime.get(me.getKey()) < 1000) {
      double[] xypos = (double[])(me.getValue());
      int x = (int)(xypos[0] * myImage.width);
      int y = (int)(xypos[1] * myImage.height);
      
      color c = get(x, y);
      
      //System.err.println("r, g, b : " + (int)red(c) + ", " + (int)green(c) + ", " + (int)blue(c));
//      dmxOutput.set(5, (int)red(c));
//      dmxOutput.set(6, (int)green(c));
//      dmxOutput.set(7, (int)blue(c));     
//   
//      dmxOutput.set(10, (int)red(c));
//      dmxOutput.set(11, (int)green(c));
//      dmxOutput.set(12, (int)blue(c));    

      drawCross(x, y, c);
    }
  }
}

void oscEvent(OscMessage theOscMessage) {  
  
  System.out.println("msg");
  
  if(theOscMessage.checkTypetag("sdd")) {
      String id = theOscMessage.get(0).stringValue();  
      double[] xypos = new double[2];
      xypos[0] = theOscMessage.get(1).doubleValue();
      xypos[1] = theOscMessage.get(2).doubleValue();
      hashmapTime.put(id, System.currentTimeMillis());
      hashmapDirection.put(id, xypos);

      return;
  }
}

void drawScene() {
  fill(m_green);
  rect(0, 0, 600, 600);
  fill(m_red);
  rect(600, 0, 600, 600);
  fill(m_blue);
  triangle(400, 400, 600, 200, 800, 400);
}

private void drawCross(int x, int y, color c) {
  color d;
  if (red(c) + green(c) + blue(c) > 765/2.0) {
    d = color(0, 0, 0);
  } else {
    d = color(255, 255, 255);
  }
  stroke(d);
  
  line(x-5, y-1, x+5, y-1);
  line(x-5, y, x+5, y);
  line(x-5, y+1, x+5, y+1);
  
  line(x-1, y-5, x-1, y+5);
  line(x, y-5, x, y+5);
  line(x+1, y-5, x+1, y+5);
  
  noStroke();
}
