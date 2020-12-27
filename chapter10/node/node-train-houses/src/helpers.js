import * as tf from '@tensorflow/tfjs-node'
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

      // Stack values
      console.log('Stacking')
      const X = tf.stack(XS)
      const Y = tf.oneHot(YS, 10)

      console.log('Images all converted to tensors:')
      console.log('X', X.shape)
      console.log('Y', Y.shape)

      // cleanup
      tf.dispose([XS])

      resolve([X, Y])
    })
  })
}
