const dfd = require('danfojs-node')
const fs = require('fs')
const prep = require('./2_featurize')

async function justWrite() {
  // Load and clean the CSV data
  const featuredData = await prep.featureData()

  const scaler = new dfd.MinMaxScaler()
  scaledData = scaler.fit(featuredData)
  scaledData.head().print()

  // Write the CSV files
  const featuredString = await scaledData.to_csv()

  fs.writeFile(
    '../../extra/featured/titanic.csv',
    featuredString,
    (err) => err && console.error(err)
  )
}

// Demonstrate
justWrite()
