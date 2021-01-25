import * as tf from '@tensorflow/tfjs-node'

const pixelShift = async (inputTensor, mutations = []) => {
  // Add 1px white padding to height and width
  const padded = inputTensor.pad(
    [
      [1, 1],
      [1, 1],
      [0, 0],
    ],
    1
  )
  const cutSize = inputTensor.shape
  for (let h = 0; h < 3; h++) {
    for (let w = 0; w < 3; w++) {
      mutations.push(padded.slice([h, w, 0], cutSize))
    }
  }
  padded.dispose()
  return mutations
}

// Creates combinations take any two from inputAarray (like Py itertools.combinations)
const combos = async (tensorArray) => {
  const startSize = tensorArray.length
  for (let i = 0; i < startSize - 1; i++) {
    for (let j = i + 1; j < startSize; j++) {
      const overlay = tf.tidy(() => {
        return tf.where(
          tf.less(tensorArray[i], tensorArray[j]),
          tensorArray[i],
          tensorArray[j]
        )      
      })
      tensorArray.push(overlay)
    }
  }
}

// Remove duplicates and stack into a 4D tensor
const consolidate = async (tensorArray) => {
  const groupedData = tf.stack(tensorArray)
  console.log("Grouped Data:", groupedData.shape)
  // Needs to switch processing to CPU for `tf.unique` on Node
  // See: https://github.com/tensorflow/tfjs/issues/4595c
  await tf.setBackend('cpu');
  const { values, _indices } = tf.unique(groupedData)
  
  tf.dispose([groupedData, _indices])
  tf.dispose(tensorArray)
  return values
}

const runAugmentation = async (aTensor) => {
  const mutes = await pixelShift(aTensor)
  await combos(mutes)
  await combos(mutes)
  return await consolidate(mutes)
  // return mutes
}

const flipTensor = (dit) => {
  return tf.tidy(() => {
    // const flip = tf.logicalNot(dit.asType("bool")).asType("float32")
    const flip = dit.sub(1).mul(-1)
    return flip
  })
}

const doStuff = async () => {
  // Tensor Array
  const inDie = require('./dice.json').data
  const imgTensor = tf.tensor(inDie[0], [12, 12, 1])
  const results = await runAugmentation(imgTensor)
  console.log('Results:', results.shape)
  const invertedResults = flipTensor(results)
  console.log('Inverted:', invertedResults.shape)
  const augResults = results.arraySync()

  tf.dispose([results, imgTensor, invertedResults])
  console.log('Memory', tf.memory().numTensors)
}

try {
  doStuff()
} catch (e) {
  console.error('ERROR', e)
}
