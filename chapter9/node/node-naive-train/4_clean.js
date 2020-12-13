const dfd = require("danfojs-node");
const prep = require("./3_combine")

async function cleanData() {
  // Combine the CSV data using './3_combine.js
  const mega = await prep.prepareData()

  /* CLEAN THE DATA
  * ------------------
  * Prune the features
  * Handle the blanks
  * Migrate to numbers
  */

  // Remove feature columns that seem less useful
  const clean = mega.drop({columns: ['Name', 'PassengerId', 'Ticket', 'Cabin']})

  // Remove all rows that have empty spots
  const onlyFull = clean.dropna()
  console.log(`After mega-clean the row-count is now ${onlyFull.shape[0]}`)

  // Handle embarked characters - convert to numbers
  const encode = new dfd.LabelEncoder()
  encode.fit(onlyFull["Embarked"])
  onlyFull['Embarked'] = encode.transform(onlyFull['Embarked'].values)
  onlyFull.head().print()

  // Handle person sex - convert to one-hot
  const sexOneHot = dfd.get_dummies(onlyFull["Sex"])
  sexOneHot.head().print()
  // Swap one column for two
  console.log(`Before shape ${onlyFull.shape}`)
  onlyFull.drop({columns: ['Sex'], axis: 1, inplace: true})
  onlyFull.addColumn({column: 'male', value: sexOneHot['male']})
  onlyFull.addColumn({column: 'female', value: sexOneHot['female']})
  // The results
  onlyFull.head().print()
  return onlyFull
}

// Demonstrate
cleanData();

// Export for use in subsequent steps
module.exports = { cleanData }
