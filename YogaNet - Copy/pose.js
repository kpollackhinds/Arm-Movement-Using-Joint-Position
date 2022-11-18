// import React, { useRef, useEffect } from "react";
// import { ReactP5Wrapper } from "react-p5-wrapper";
// import * as math from "mathjs";
// import teapotURL from "./assets/teapot.obj";
// import greenColor from "./assets/green.png";
// import Helper from "./functions/helper";
// import ExportCSV from "./functions/exportToCSV";
// import * as Holistic from "@mediapipe/holistic/holistic";
// // mediapipe camrea tools
// import * as Camera from "@mediapipe/camera_utils/camera_utils";
// // p5.js font
// import Inconsolata from "./font/Inconsolata-Black.otf";

//limb motion checkbox -- currently set to true because checkbox not created yet
let limbMotionChecked = true;
let limbSelected = {
  spine: {
    selected: false,
    index1: [11, 12],
    index2: [23, 24],
    color: [128, 0, 0],
  },
  ruarm: { selected: false, index1: 12, index2: 14, color: [0, 0, 255] },
  rlarm: { selected: false, index1: 14, index2: 16, color: [0, 255, 255] },
  luarm: { selected: false, index1: 11, index2: 13, color: [139, 0, 139] },
  llarm: { selected: false, index1: 13, index2: 15, color: [255, 0, 255] },
  ruleg: { selected: false, index1: 24, index2: 26, color: [139, 69, 19] },
  rlleg: { selected: false, index1: 26, index2: 28, color: [244, 164, 96] },
  luleg: { selected: false, index1: 23, index2: 25, color: [255, 69, 0] },
  llleg: { selected: false, index1: 25, index2: 27, color: [240, 230, 140] },
};

// joint motion cb
let jointMotionChecked = false;
let jointSelected = {
  rshoulder: { selected: false, index: 12, color: [255, 140, 0] },
  relbow: { selected: false, index: 14, color: [128, 128, 0] },
  rwrist: { selected: false, index: 16, color: [255, 20, 147] },
  lshoulder: { selected: false, index: 11, color: [85, 107, 47] },
  lelbow: { selected: false, index: 13, color: [0, 255, 0] },
  lwrist: { selected: false, index: 15, color: [0, 250, 154] },
  rhip: { selected: false, index: 24, color: [0, 0, 255] },
  rknee: { selected: false, index: 26, color: [138, 43, 226] },
  rankle: { selected: false, index: 28, color: [106, 90, 205] },
  lhip: { selected: false, index: 23, color: [255, 0, 0] },
  lknee: { selected: false, index: 25, color: [244, 164, 96] },
  lankle: { selected: false, index: 27, color: [112, 128, 144] },
};

let videoElement = document.getElementById("input_video");
let rightArmAngle;
let leftArmAngle;
let rightLegAngle;
let leftLegAngle;
let rightUpperArmAngle;
let left

function onResults(results) {
  // pose detection
  if (results.poseWorldLandmarks) {
    const pose = results.poseLandmarks;
    rightArmAngle = getAngle(pose[12].x, pose[12].y, pose[14].x, pose[14].y, pose[16].x, pose[16].y);
    leftArmAngle = getAngle(pose[11].x, pose[11].y, pose[13].x, pose[13].y, pose[15].x, pose[15].y);
    rightLegAngle = getAngle(pose[24].x, pose[24].y, pose[26].x, pose[26].y, pose[28].x, pose[28].y);
    leftLegAngle = getAngle(pose[23].x, pose[23].y, pose[25].x, pose[25].y, pose[27].x, pose[27].y);

    document.getElementById('output').innerHTML = rightArmAngle;
    let inputs = []
    for(let i=0; i<33; i++){
        inputs.push(results.poseWorldLandmarks[i].x)
        inputs.push(results.poseWorldLandmarks[i].y)
        inputs.push(results.poseWorldLandmarks[i].z)
    }
    // if(classification) brain.classify(inputs, gotResult);
    // if(poseLabel) console.log(poseLabel)
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
  // console.log('1. Angle:', angle1);  

  
  // vec_a = [(x2-x1), (y2-y1)];
  // vec_b = [(x3-x2), (y3-y2)];

  // dotProduct = (vec_a[0] * vec_b[0])+(vec_a[1] * vec_b[1]);

  // magn_a = Math.sqrt(Math.pow(vec_a[0],2) + Math.pow(vec_a[1],2));
  // magn_b = Math.sqrt(Math.pow(vec_b[0],2) + Math.pow(vec_b[1],2));

  return (Math.acos(dotProduct / (magn_a*magn_b)) * 180 / Math.PI);
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
  width: 1280,
  height: 720,
});
camera.start();
// trained model
let opt = {
    inputs: 99,
    outputs: 1,
    task: "classification",
    debug: true,
  };

let brain = ml5.neuralNetwork(opt);

const modelInfo = {
model: "model/model.json",
metadata: "model/model_meta.json",
weights: "model/model.weights.bin",
};

brain.load(modelInfo, brainLoaded);

let classification = false;
let poseLabel;

function brainLoaded() {
    console.log('classification ready')
    classification = true; 
}

// function gotResult(error, results) {
//     if (results[0].confidence > 0.75) {
//         poseLabel = results[0].label
//     }
// }

