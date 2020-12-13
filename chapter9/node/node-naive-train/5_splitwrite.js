const dfd = require("danfojs-node");
const fs = require('fs');
const prep = require("./4_clean")

async function splitWrite() {
  // Load and clean the CSV data
  const onlyFull = await prep.cleanData()

  // 500 random to training
  const newTrain = onlyFull.sample(500)
  console.log(`newTrain row count: ${newTrain.shape[0]}`)
  // The rest to testing (drop via row index)
  const newTest = onlyFull.drop({index: newTrain.index, axis: 0})
  console.log(`newTest row count: ${newTest.shape[0]}`)

  // Write the CSV files
  const trainString = await newTrain.to_csv()
  const testString = await newTest.to_csv()
  fs.writeFile('../../extra/cleaned/newTrain.csv', trainString, (err) =>
    err && console.error(err)
  )
  fs.writeFile('../../extra/cleaned/newTest.csv', testString, (err) =>
    err && console.error(err)
  )
}

// Demonstrate
splitWrite();


