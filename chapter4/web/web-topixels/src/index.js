import * as tf from "@tensorflow/tfjs";

const bigMess = tf.randomUniform([400, 400, 3]);
const myCanvas = document.getElementById("randomness");
tf.browser.toPixels(bigMess, myCanvas).then(() => {
  bigMess.dispose();
  console.log("Make sure we cleaned up", tf.memory().numTensors);
});
