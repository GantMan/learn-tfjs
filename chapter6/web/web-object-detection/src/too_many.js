import * as tf from "@tensorflow/tfjs";
import "core-js/stable";
import "regenerator-runtime/runtime";

async function performDetections() {
  await tf.ready();
  const modelPath =
    "https://tfhub.dev/tensorflow/tfjs-model/ssd_mobilenet_v2/1/default/1";

  const model = await tf.loadGraphModel(modelPath, { fromTFHub: true });
  const mysteryImage = document.getElementById("mystery");
  const myTensor = tf.browser.fromPixels(mysteryImage);
  // SSD Mobilenet single batch
  const readyfied = tf.expandDims(myTensor, 0);
  const results = await model.executeAsync(readyfied);
  const boxes = await results[1].squeeze().array();

  // Prep Canvas
  const detection = document.getElementById("detection");
  const ctx = detection.getContext("2d");
  const imgWidth = mysteryImage.width;
  const imgHeight = mysteryImage.height;
  detection.width = imgWidth;
  detection.height = imgHeight;

  boxes.forEach((box, idx) => {
    ctx.strokeStyle = "#0F0";
    ctx.lineWidth = 1;
    const startY = box[0] * imgHeight;
    const startX = box[1] * imgWidth;
    const height = (box[2] - box[0]) * imgHeight;
    const width = (box[3] - box[1]) * imgWidth;
    ctx.strokeRect(startX, startY, width, height);
  });
}

performDetections();
