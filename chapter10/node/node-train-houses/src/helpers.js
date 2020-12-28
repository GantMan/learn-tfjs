import * as tf from '@tensorflow/tfjs-node'
// import * as tf from '@tensorflow/tfjs-node-gpu'
import * as fs from 'fs'
import { default as glob } from 'glob'

function encodeDir(filePath) {
  if (filePath.includes('bird')) return 0
  if (filePath.includes('lion')) return 1
  if (filePath.includes('owl')) return 2
  if (filePath.includes('parrot')) return 3
  if (filePath.includes('raccoon')) return 4
  if (filePath.includes('skull')) return 5
  if (filePath.includes('snail')) return 6
  if (filePath.includes('snake')) return 7
  if (filePath.includes('squirrel')) return 8
  if (filePath.includes('tiger')) return 9

  // Should never get here
  console.error('Unrecognized folder')
  process.exit(1)
}

function shuffleCombo(array, array2) {
  let counter = array.length
  console.assert(array.length === array2.length)
  let temp, temp2
  let index = 0
  // While there are elements in the array
  while (counter > 0) {
    // Pick a random index
    index = (Math.random() * counter) | 0
    // Decrease counter by 1
    counter--
    // And swap the last element with it
    temp = array[counter]
    temp2 = array2[counter]
    array[counter] = array[index]
    array2[counter] = array2[index]
    array[index] = temp
    array2[index] = temp2
  }
}

export function folderToTensors() {
  return new Promise((resolve, reject) => {
    const FILE_PATH = 'files'
    // create stack in JS
    const XS = []
    const YS = []

    // Read images
    console.log('Identifying PNG List')
    glob('files/**/*.png', (err, files) => {
      if (err) {
        console.error('Failed to access PNG files', err)
        reject()
        process.exit(1)
      }

      console.log(`${files.length} Files Found`)

      console.log('Now converting to tensors')
      files.forEach((file) => {
        const imageData = fs.readFileSync(file)
        const answer = encodeDir(file)
        const imageTensor = tf.node.decodeImage(imageData, 1)

        // Store in memory
        YS.push(answer)
        XS.push(imageTensor)
      })

      // Shuffle the data (keep XS[n] === YS[n])
      shuffleCombo(XS, YS)

      // Stack values
      console.log('Stacking')
      const X = tf.stack(XS)
      const Y = tf.oneHot(YS, 10)

      console.log('Images all converted to tensors:')
      console.log('X', X.shape)
      console.log('Y', Y.shape)

      // Normalize X to values 0 - 1
      const XNORM = X.div(255)
      // cleanup
      tf.dispose([XS, X])

      resolve([XNORM, Y])
    })
  })
}

function printSign(val) {
  console.log(`
  ╔════════════╗
  ║   SAVING   ║
  ║   %${(val * 100).toFixed(2)}   ║
  ╚════════════╝
  `)
}

export async function bestValidationSave(model, savePath, best) {
  return {
    onEpochEnd: async (_epoch, logs) => {
      if (logs.val_acc > best) {
        printSign(logs.val_acc)
        model.save(savePath)
        best = logs.val_acc
      }
    },
  }
}
