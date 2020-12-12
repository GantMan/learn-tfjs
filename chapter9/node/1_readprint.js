const dfd = require("danfojs-node");

async function prepareData() {
  const df = await dfd.read_csv("file://../extra/titanic data/train.csv");
  // Print the columns
  df.head().print();
  // Describe the DataFrame
  df.describe().print();
}

prepareData();
