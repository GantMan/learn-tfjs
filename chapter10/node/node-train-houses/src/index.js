import * as tf from '@tensorflow/tfjs-node'
import getModel from './getModel'
import { folderToTensors } from './helpers'

async function doTraining() {
  try {
    // Read images
    const [X, Y] = await folderToTensors()

    // Create layers model
    const model = getModel()

    // Train
    await model.fit(X, Y, {
      batchSize: 256,
      validationSplit: 0.1,
      epochs: 20,
      // shuffle: true,
    })

    // Cleanup!
    tf.dispose([X, Y, model])
    console.log('Tensors in memory', tf.memory().numTensors)
  } catch (e) {
    console.log(e)
  }
}

doTraining()
