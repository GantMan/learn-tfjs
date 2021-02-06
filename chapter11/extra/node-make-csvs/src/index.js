// Try to use GPU where possible
import * as tf from '@tensorflow/tfjs-node'
const dfd = require('danfojs-node')
// import * as tf from '@tensorflow/tfjs-node-gpu'
import { folderToTensors } from './helpers'

async function buildCSVs() {
  // Read images
  const [X, Y] = await folderToTensors()

  // Write labels file
  let df = new dfd.DataFrame(Y, {
    columns: ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king'],
  })
  df.to_csv('chess_labels.csv')
  Y.dispose()

  let dfdata = new dfd.DataFrame(X)
  X.dispose()
  
  let dfdataRound = dfdata.round()
  dfdataRound.head().print()
  
  dfdataRound.to_csv('chess_images.csv')

  console.log('BACK Y', df.shape)
  console.log('BACK X', dfdataRound.shape)

  // Cleanup!
  console.log('Tensors in memory', tf.memory().numTensors)
}

buildCSVs()
