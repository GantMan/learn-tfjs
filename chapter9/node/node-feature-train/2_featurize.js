const dfd = require('danfojs-node')
const prep = require('./1_combine')

function ageToBucket(x) {
  if (x < 10) {
    return 0
  } else if (x < 40) {
    return 1
  } else {
    return 2
  }
}

async function featureData() {
  // Combine the CSV data using './3_combine.js
  const mega = await prep.prepareData()

  // Handle person sex - convert to one-hot
  const sexOneHot = dfd.get_dummies(mega['Sex'])
  sexOneHot.head().print()
  // Swap one column for two
  console.log(`Before shape ${mega.shape}`)
  mega.drop({ columns: ['Sex'], axis: 1, inplace: true })
  mega.addColumn({ column: 'male', value: sexOneHot['0'] })
  mega.addColumn({ column: 'female', value: sexOneHot['1'] })

  // Create Age buckets
  ageBuckets = mega['Age'].apply(ageToBucket)
  mega.addColumn({ column: 'Age_bucket', value: ageBuckets })

  // The results
  mega.head().print()
  return mega
}

// Demonstrate
featureData()

// Export for use in subsequent steps
module.exports = { featureData }
