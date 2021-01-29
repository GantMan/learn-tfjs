import * as tf from '@tensorflow/tfjs-node'
import { shuffleCombo } from './helper'

const inputShape = [12, 12, 1]
const epochs = 20
const testSplit = 0.05
const diceData = require('./dice_data.json')

function prepData() {
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

  return [trainX, trainY, testX, testY]
}

const trainModel = async (data) => {
  const model = tf.sequential()

  model.add(tf.layers.flatten({ inputShape }))

  model.add(
    tf.layers.dense({
      units: 64,
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

  await model.fit(data[0], data[1], {
    epochs,
    validationSplit: 0.1,
    shuffle: true,
  })

  console.log('Done Training')
  return model
}

async function evaluateResults(model, data) {
  const result = model.evaluate(data[2], data[3])
  console.log('Test Loss', result[0].dataSync())
  console.log('Test Accuracy', result[1].dataSync())
  tf.dispose(result)
}

async function makeModel() {
  const data = await prepData()
  const model = await trainModel(data)
  evaluateResults(model, data)
  model.save('file://./dice-model')
}

makeModel()
