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
  await scaledData.to_csv('../../extra/featured/titanic.csv')
  console.log('Featured file written')
}

// Demonstrate
justWrite()
