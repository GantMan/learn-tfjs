const dfd = require('danfojs-node')

async function prepareData() {
  // Train data
  const df = await dfd.read_csv('file://../../extra/cleaned/newTrain.csv')
  console.log('Train Size', df.shape[0])
  const dft = await dfd.read_csv('file://../../extra/cleaned/newTest.csv')
  console.log('Test Size', dft.shape[0])

  const mega = dfd.concat({ df_list: [df, dft], axis: 0 })
  console.log('The CSVs Combined')
  mega.describe().print()

  return mega
}

// Demonstrate
prepareData()

// Export for use in subsequent steps
module.exports = { prepareData }
