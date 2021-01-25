import * as tf from '@tensorflow/tfjs-node'
import { shuffleCombo } from './helper'

const inputShape = [12, 12, 1]
const epochs = 10
const testSplit = 0.05
const diceData = require('./dice_data.json')

// Wrap in a tidy for memory
const [trainX, trainY, testX, testY] = tf.tidy(() => {
  // Build a stacked tensor from JSON
  const diceImages = [].concat(
    diceData['0'],
    diceData['1'],
    diceData['2'],
    diceData['3'],
    diceData['4'],
    diceData['5'],
    diceData['6'],
    diceData['7'],
    diceData['8'],
    diceData['inverted0'],
    diceData['inverted1'],
    diceData['inverted2'],
    diceData['inverted3'],
    diceData['inverted4'],
    diceData['inverted5'],
    diceData['inverted6'],
    diceData['inverted7'],
    diceData['inverted8']
  )

  // Now the answers to their corresponding index
  const answers = [].concat(
    new Array(diceData['0'].length).fill(0),
    new Array(diceData['1'].length).fill(1),
    new Array(diceData['2'].length).fill(2),
    new Array(diceData['3'].length).fill(3),
    new Array(diceData['4'].length).fill(4),
    new Array(diceData['5'].length).fill(5),
    new Array(diceData['6'].length).fill(6),
    new Array(diceData['7'].length).fill(7),
    new Array(diceData['8'].length).fill(8),
    new Array(diceData['inverted0'].length).fill(9),
    new Array(diceData['inverted1'].length).fill(10),
    new Array(diceData['inverted2'].length).fill(11),
    new Array(diceData['inverted3'].length).fill(12),
    new Array(diceData['inverted4'].length).fill(13),
    new Array(diceData['inverted5'].length).fill(14),
    new Array(diceData['inverted6'].length).fill(15),
    new Array(diceData['inverted7'].length).fill(16),
    new Array(diceData['inverted8'].length).fill(17)
  )

  // Randomize & Split
  shuffleCombo(diceImages, answers)
  // Group into train/test split
  const testCount = parseInt(diceImages.length * testSplit)
  const trainCount = diceImages.length - testCount
  const testImgData = diceImages.slice(trainCount)
  const testAnswerData = answers.slice(trainCount)
  diceImages.splice(trainCount)
  answers.splice(trainCount)

  // Convert to tensors
  const numOptions = Object.keys(diceData).length
  const trainX = tf.tensor(diceImages).expandDims(3)
  const trainY = tf.oneHot(answers, numOptions)
  const testX = tf.tensor(testImgData).expandDims(3)
  const testY = tf.oneHot(testAnswerData, numOptions)

  return [trainX, trainY, testX, testY]
})

console.log('trainX ', trainX.shape)
console.log('trainY ', trainY.shape)
console.log('testX ', testX.shape)
console.log('testY ', testY.shape)

const trainModel = async () => {
  const model = tf.sequential()

  model.add(tf.layers.flatten({ inputShape }))

  model.add(
    tf.layers.dense({
      units: 64,
      inputShape: inputShape,
      activation: 'relu',
    })
  )
  model.add(
    tf.layers.dense({
      units: 8,
      activation: 'relu',
    })
  )
  model.add(
    tf.layers.dense({
      units: 9,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax',
    })
  )
  const learningRate = 0.005
  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  })

  // Make loss callback
  const printCallback = {
    onEpochEnd: (epoch, log) => {
      console.log(`${epoch + 1} of ${epochs}`, log)
    },
  }

  await model.fit(trainX, trainY, {
    epochs,
    shuffle: true,
    batchSize: 32,
    callbacks: printCallback,
  })

  console.log('Done')
}
