const dfd = require("danfojs-node");

async function prepareData() {
  // Train data
  const df = await dfd.read_csv("file://../../extra/titanic data/train.csv");
  console.log("Train Size", df.shape[0]);
  const dft = await dfd.read_csv("file://../../extra/titanic data/test.csv");
  console.log("Test Size", dft.shape[0]);

  // Combined they are 1309 values
  const mega = dfd.concat({ df_list: [df, dft], axis: 0 });
  console.log("The CSVs Combined");
  mega.describe().print();

  return mega;
}

// Demonstrate
prepareData();

// Export for use in subsequent steps
module.exports = { prepareData };
