#include <SoftwareSerial.h>
#include <Servo.h>


const byte rxPin = 2;
const byte txPin = 3;
SoftwareSerial HM10(rxPin, txPin); // RX = 2, TX = 3

char data;
String stringData = "";

String message = "";
String angleUpperServo = "";
String angleLowerServo = "";



Servo upperServo;
Servo lowerServo;


int pos = 0;
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  HM10.begin(9600);
  Serial.println("set HM10 at 9600 baud rate");

  pinMode(13, OUTPUT);
  digitalWrite(13, LOW);

  upperServo.attach(10);
  lowerServo.attach(9);

  upperServo.write(5);
  lowerServo.write(5);

  
}

void loop() {
  //.available() gets number of bytes available for reading 
  if(HM10.available() > 0 && stringData != "-"){
    digitalWrite(13, HIGH);
    data = HM10.read();  //reads data waiting in serial recieve buffer
    stringData = String(data);
//    Serial.print(stringData);
    message = message + stringData;
    
  } else if(message != "" && stringData == "-" ){ //mesage is over when last character is '-'
//    Serial.println(message);
    adjustAngle(message);
    message = "";
    stringData = "";   
  }
}


//function to process the recieved angle data and write servo to that angle
void adjustAngle(String angles){
  Serial.println("yes");
  angleUpperServo = message.substring(0, message.indexOf("&"));
  angleLowerServo = message.substring((message.indexOf("&")+1), message.indexOf("-"));

//  Serial.println(angleLowerServo + "--" + angleUpperServo);
  Serial.println(angleLowerServo);

//  Serial.println(angleUpperServo);

  if(angleUpperServo != " "){
    upperServo.write(angleUpperServo.toInt());
  }
//  delay(100);
  if(angleLowerServo != " "){
    lowerServo.write(180 - angleLowerServo.toInt());
  }
  
}
