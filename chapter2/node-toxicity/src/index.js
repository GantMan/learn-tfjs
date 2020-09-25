import '@tensorflow/tfjs-node'
import * as toxicity from '@tensorflow-models/toxicity'

// minimum positive prediction confidence
// If this isn't passed, the default is 0.85
const threshold = 0.5

toxicity.load(threshold).then((model) => {
  const sentences = ['You are a poopy head!', 'I like turtles', 'Shut up!']

  model.classify(sentences).then((predictions) => {
    // semi-pretty-print results
    console.log(JSON.stringify(predictions, null, 2))
  })
})
