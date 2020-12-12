const dfd = require("danfojs-node");

async function prepareData() {
  const df = await dfd.read_csv("file://../extra/titanic data/train.csv");
  // Count of empty spots
  empty_spots = df.isna().sum();
  empty_spots.print();
  empty_rate = empty_spots.div(df.isna().count());
  empty_rate.print();
}

prepareData();
