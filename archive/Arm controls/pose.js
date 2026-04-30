let width = 640,
  height = 480;

const serviceUuid = "0000ffe0-0000-1000-8000-00805f9b34fb"; //check what uuid is for HM-10 Bluetooth Module
let myBLE;
let myCharacteristic;   

let buttonValue = 0;
let input;

let counter = 0;

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function setup() {
  //p5 setup stuff
  createCanvas(width, height);
  video = createCapture(VIDEO);
  video.size(width/2, height);
  video.hide();

  // WRITE TO BLE BUTTON -------
  // input = createInput();
  // input.position(15, 480);

  // const writeButton = createButton('Write');
  // writeButton.position(input.x + input.width + 15, 100);
  // writeButton.mousePressed(writeToBle2);
  //---------------

  myBLE = new p5ble();
  background("#FFF");
  const connectButton = createButton('Connect and Start Notifications');
  connectButton.mousePressed(connectAndStartNotify);
}

function connectAndStartNotify() {
  // Connect to a device by passing the service UUID
  myBLE.connect(serviceUuid, gotCharacteristics);
}

// A function that will be called once got characteristics
function gotCharacteristics(error, characteristics) {
  if (error) console.log('error: ', error);
  console.log('characteristics: ', characteristics);
  // Set the first characteristic as myCharacteristic
  myCharacteristic = characteristics[0];
}

// // A function that will be called once got values
// function gotValue(error, value) {
//   if (error) console.log('error: ', error);
//   console.log('value: ', value);
//   myValue = value;
//   // After getting a value, call p5ble.read() again to get the value again
//   myBLE.read(myCharacteristic, gotValue);
// }

function writeToBle(message) {
  // const inputValue = input.value();
  // Write the value of the input to the myCharacteristic
  myBLE.write(myCharacteristic, message);
}

function draw() {
  background(250);
  // circle(width/2, 10, 20);
  image(video, width/2, 0, width/2, height/2);
  // text(myValue, 100, 100);
    // noStroke();
    
  //  if(buttonValue>0){
  //    fill(color(200, 200, 200));
  //  }else{
  //    fill(color(100, 200, 200));
  //  }
  //  rect(15, 40, 60, 60);
  
}
let videoElement = document.getElementById("input_video");
let rightArmAngle;
let leftArmAngle;
let rightLegAngle;
let leftLegAngle;
let rightUpperArmAngle;
let currentRightArmAngle = 0;
let currentRUpperArmAngle = 0;
let left

function onResults(results) {
  // pose detection
  if (myBLE.isConnected() && results.poseWorldLandmarks && counter % 10 == 0) {
    if (counter == 100) counter = 0;  //Stops code from sending to much data to bluetooth at once
    const pose = results.poseLandmarks;
    rightArmAngle = getAngle(pose[12].x, pose[12].y, pose[14].x, pose[14].y, pose[16].x, pose[16].y);
    rightUpperArmAngle = getAngle(pose[14].x, pose[14].y, (pose[11].x + pose[12].x)/2, (pose[11].y + pose[12].y)/2, (pose[23].x + pose[24].x)/2, (pose[23].y + pose[24].y)/2);
    // document.getElementById('output').innerHTML = rightArmAngle;

    //parseInt essentially makes the float values an int
    //check if the angle has changed by at least five degees.
    if(Math.abs(rightArmAngle-currentRightArmAngle) > 5 && Math.abs(rightUpperArmAngle-currentRUpperArmAngle) > 5 ) {
      console.log(parseInt(rightUpperArmAngle).toString() + "&" + parseInt(rightArmAngle).toString() + "-");
      writeToBle(parseInt(rightUpperArmAngle).toString() + "&" + parseInt(rightArmAngle).toString() + "-");
      currentRightArmAngle = rightArmAngle;
      currentRUpperRightArmAngle = rightUpperArmAngle;
    }

    else if(Math.abs(rightArmAngle-currentRightArmAngle) > 5) {
      console.log(" " + "&" + parseInt(rightArmAngle).toString() + "-");
      writeToBle(" " + "&" + parseInt(rightArmAngle).toString() + "-");
      currentRightArmAngle = rightArmAngle;
    }

    else if(Math.abs(rightUpperArmAngle-currentRUpperArmAngle) > 5 ) {
      console.log(parseInt(rightUpperArmAngle).toString() + "&" + " " + "-");
      writeToBle(parseInt(rightUpperArmAngle).toString() + "&" + " " + "-");
      currentRUpperRightArmAngle = rightUpperArmAngle;
    }
  }
}
const drawResults = (results, time) => {
  const canvasElement = output_canvas.current;
  const canvasCtx = canvasElement.getContext("2d");
  // Calc FPS
  FPS = math.floor(1000 / (time - previousFrameTime));
  previousFrameTime = time;

  if (results.poseLandmarks) poses = results.poseLandmarks;

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
};

function getAngle(x1, y1, x2, y2, x3, y3){

  let a = {x: (x1-x2), y: (y1-y2), z: 0};
  let b = {x: (x3-x2), y: (y3-y2), z: 0}; // set z != 0 for 3D

  let dot = (p1, p2)=>  p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
  let magSq = ({x, y, z}) => x ** 2 + y ** 2 + z ** 2;

  return( Math.acos(dot(a, b) / Math.sqrt(magSq(a) * magSq(b)))* 180 / Math.PI);
}

let pose = new Pose({
  locateFile: (file) => {
    //console.log(file);
    return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
  },
});
pose.setOptions({
  modelComplexity: 0,
  smoothLandmarks: true,
  enableSegmentation: true,
  smoothSegmentation: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});
pose.onResults(onResults);

let camera = new Camera(videoElement, {
  onFrame: async () => {
    await pose.send({ image: videoElement });
  },
  width: 640,
  height: 480,
});
camera.start();

// // trained model
// let opt = {
//     inputs: 99,
//     outputs: 1,
//     task: "classification",
//     debug: true,
//   };

// let brain = ml5.neuralNetwork(opt);

// const modelInfo = {
// model: "model/model.json",
// metadata: "model/model_meta.json",
// weights: "model/model.weights.bin",
// };

// brain.load(modelInfo, brainLoaded);

// let classification = false;
// let poseLabel;

// function brainLoaded() {
//     console.log('classification ready')
//     classification = true; 
// }

// function gotResult(error, results) {
//     if (results[0].confidence > 0.75) {
//         poseLabel = results[0].label
//     }
// }

