const dfd = require('danfojs-node')
// const tf = require("@tensorflow/tfjs"); // Use if having issues with tfjs-node
const tf = require('@tensorflow/tfjs-node')

const modelMaker = require('../node-shared-model/getModel')

async function trainModel() {
  // Get featured data
  const df = await dfd.read_csv('file://../../extra/featured/titanic.csv')

  // Split train into X/Y
  const trainX = df.iloc({ columns: [`1:`] }).tensor
  const trainY = df['0'].tensor

  // Make and train
  const inputShape = [trainX.shape[1]]
  const model = modelMaker.getModel(tf, inputShape)
  await model.fit(trainX, trainY, {
    batchSize: 32,
    epochs: 100,
    validationSplit: 0.2, // Asking the model to save 20% for validation on the fly
    // callbacks: modelMaker.callbacks, // use when using tfjs instead of tfjs-node
  })
}

// Demonstrate
trainModel()
