const dfd = require('danfojs-node')
// const tf = require("@tensorflow/tfjs"); // Use if having issues with tfjs-node
const tf = require('@tensorflow/tfjs-node')

const modelMaker = require('../node-shared-model/getModel')

async function trainModel() {
  // Get cleaned data
  const df = await dfd.read_csv('file://../../extra/cleaned/newTrain.csv')
  console.log('Train Size', df.shape[0])
  const dft = await dfd.read_csv('file://../../extra/cleaned/newTest.csv')
  console.log('Test Size', dft.shape[0])

  // Split train into X/Y
  const trainX = df.iloc({ columns: [`1:`] }).tensor
  const trainY = df['Survived'].tensor

  // Split test into X/Y
  const testX = dft.iloc({ columns: [`1:`] }).tensor
  const testY = dft['Survived'].tensor

  // Make and train
  const inputShape = [df.shape[1] - 1]
  const model = modelMaker.getModel(tf, inputShape)
  await model.fit(trainX, trainY, {
    batchSize: 32,
    epochs: 100,
    validationData: [testX, testY],
    // callbacks: modelMaker.callbacks, // use when using tfjs instead of tfjs-node
  })
}

// Demonstrate
trainModel()
