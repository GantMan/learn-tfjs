import * as tf from "@tensorflow/tfjs-node";

const pixelShift = async (inputTensor, mutations = []) => {
  // Add 1px white padding to height and width
  const padded = inputTensor.pad(
    [
      [1, 1],
      [1, 1],
      [0, 0]
    ],
    255
  );
  const cutSize = inputTensor.shape;
  for (let h = 0; h < 3; h++) {
    for (let w = 0; w < 3; w++) {
      mutations.push(padded.slice([h, w, 0], cutSize));
    }
  }
  padded.dispose();
  return mutations;
};

// Creates combinations take any two from inputAarray (like Py itertools.combinations)
const combos = async (tensorArray) => {
  const arrCopy = [...tensorArray];
  const startSize = arrCopy.length;
  for (let i = 0; i < startSize - 1; i++) {
    for (let j = i + 1; j < startSize; j++) {
      const overlay = tf.where(
        tf.less(arrCopy[i], arrCopy[j]),
        arrCopy[i],
        arrCopy[j]
      );
      tensorArray.push(overlay);
    }
  }
};

// Remove duplicates and stack into a 4D tensor
const consolidate = async (tensorArray) => {
  const groupedData = tf.stack(tensorArray);
  const { values } = tf.unique(groupedData);
  groupedData.dispose();
  return values;
};

const runData = async (aTensor) => {
  const mutes = await pixelShift(aTensor);
  await combos(mutes);
  await combos(mutes);
  return await consolidate(mutes);
};

const flipTensor = (dit) => {
  // const flip = tf.logicalNot(dit.asType("bool")).asType("float32")
  const flip = dit.sub(1).mul(-1)
  return flip
}

const doStuff = async () => {
  // Tensor Array
  const inTensor = require("./inTensor.json").data;
  // some image here
  const imgTensor = tf.tensor(inTensor[0]);
  const results = await runData(imgTensor);
  const augResults = results.reshape([results.shape[0], 12, 12]).arraySync();

  console.log(augResults.shape)
  tf.dispose([augResults, results, imgTensor])
  console.log(tf.memory().numTensors)
};

doStuff();
