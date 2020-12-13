const dfd = require("danfojs-node");

async function readCSV() {
  const df = await dfd.read_csv("file://../../extra/titanic data/train.csv");
  // Print the columns
  df.head().print();
  // Describe the DataFrame
  df.describe().print();
}

// Demonstrate
readCSV();
