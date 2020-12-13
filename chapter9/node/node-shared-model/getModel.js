function getModel(tf, inputShape) {
  const model = tf.sequential()

  model.add(
    tf.layers.dense({
      inputShape,
      units: 120,
      activation: 'relu',
      kernelInitializer: 'heNormal',
    })
  )
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }))
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }))
  model.add(
    tf.layers.dense({
      units: 1,
      activation: 'sigmoid',
    })
  )

  model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  })

  return model
}

// Export for use in subsequent steps
module.exports = { getModel }
