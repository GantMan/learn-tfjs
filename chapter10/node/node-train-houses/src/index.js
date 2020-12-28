// Try to use GPU where possible
import * as tf from '@tensorflow/tfjs-node'
// import * as tf from '@tensorflow/tfjs-node-gpu'
import getModel from './getModel'
import { folderToTensors, bestValidationSave } from './helpers'

async function doTraining() {
  // Read images
  const [X, Y] = await folderToTensors()

  // Create layers model
  const model = getModel()

  let best = 0
  // Train
  await model.fit(X, Y, {
    batchSize: 256,
    validationSplit: 0.1,
    epochs: 20,
    shuffle: true,
    callbacks: bestValidationSave(
      model,
      'file://model_result/sorting_hat',
      best
    ),
  })

  // Cleanup!
  tf.dispose([X, Y, model])
  console.log('Tensors in memory', tf.memory().numTensors)
}

doTraining()
