const dfd = require('danfojs-node')
const prep = require('./4_clean')

async function splitWrite() {
  // Load and clean the CSV data
  const onlyFull = await prep.cleanData()

  // 500 random to training
  const newTrain = onlyFull.sample(800)
  console.log(`newTrain row count: ${newTrain.shape[0]}`)
  // The rest to testing (drop via row index)
  const newTest = onlyFull.drop({ index: newTrain.index, axis: 0 })
  console.log(`newTest row count: ${newTest.shape[0]}`)

  // Write the CSV files
  await newTrain.to_csv('../../extra/cleaned/newTrain.csv')
  await newTest.to_csv('../../extra/cleaned/newTest.csv')
  console.log('Files written!')
}

// Demonstrate
splitWrite()
