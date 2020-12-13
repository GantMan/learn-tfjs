function getModel(tf, inputShape) {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape,
      units: 120,
      activation: "relu",
      kernelInitializer: "heNormal",
    })
  );
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 32, activation: "relu" }));
  model.add(
    tf.layers.dense({
      units: 1,
      activation: "sigmoid",
    })
  );

  model.compile({
    optimizer: "adam",
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

// Useful if you need to use tfjs instead of tfjs-node
const callbacks = {
  onEpochEnd: async (epoch, logs) => {
    console.log(`
      EPOCH (${epoch + 1}): 
        Train Accuracy: ${(logs.acc * 100).toFixed(2)},
        Val Accuracy:  ${(logs.val_acc * 100).toFixed(2)}
    `);
  },
};

// Export for use in subsequent steps
module.exports = { getModel, callbacks };
