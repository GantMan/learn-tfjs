import * as tf from '@tensorflow/tfjs-node'
import * as fs from 'fs'
import * as path from 'path'

const FILE_PATH = 'files'
const cakeImagePath = path.join(FILE_PATH, 'cake.jpg')
const gantCakePath = path.join(FILE_PATH, 'gantcake.gif')
const cakeImage = fs.readFileSync(cakeImagePath)
const gantCake = fs.readFileSync(gantCakePath)

tf.tidy(() => {
  const cakeTensor = tf.node.decodeImage(cakeImage)
  console.log(`Success: local file to a ${cakeTensor.shape} tensor`)

  const cakeBWTensor = tf.node.decodeImage(cakeImage, 1)
  console.log(`Success: local file to a ${cakeBWTensor.shape} tensor`)

  const gantCakeTensor = tf.node.decodeImage(gantCake, 3, 'int32', true)
  console.log(`Success: local file to a ${gantCakeTensor.shape} tensor`)
})

console.log(tf.memory().numTensors)
