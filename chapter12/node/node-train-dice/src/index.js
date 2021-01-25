import * as tf from '@tensorflow/tfjs-node'

const inputShape = [12, 12, 1]
const epochs = 10
const diceData = require('./dice_data.json').dice

// Wrap in a tidy for memory
const [stackedX, stackedY] = tf.tidy(() => {
  // Build a stacked tensor from JSON
  const xs = tf
    .concat([
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
      diceData['inverted8'],
    ])
    .expandDims(3)

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
  const ys = tf.oneHot(answers, diceData.length)

  return [xs, ys]
})

const testModel = (model, img2Test) => {
  const img = new Image()
  img.crossOrigin = 'anonymous'
  img.src = img2Test
  img.onload = async () => {
    const imgTensor = tf.browser.fromPixels(img, 1).expandDims()

    const tensorResults = model.predict(imgTensor)
    const results = tensorResults.arraySync()

    console.log(`${img2Test} returned ${results}`)
  }
}

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

  await model.fit(stackedX, stackedY, {
    epochs,
    shuffle: true,
    batchSize: 32,
    callbacks: printCallback,
  })

  console.log('Done')
}
