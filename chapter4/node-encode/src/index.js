import * as tf from "@tensorflow/tfjs-node";

const fs = require("fs");
const bigMess = tf.randomUniform([400, 400, 3], 0, 255);
tf.node.encodeJpeg(bigMess, "rgb").then((f) => {
  // console.log(f);
  fs.writeFileSync("out.jpg", f);
  console.log("File written");
});
