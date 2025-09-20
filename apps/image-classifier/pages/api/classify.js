import sharp from 'sharp';
import fs from 'fs';
import formidable from 'formidable';
import * as tf from '@tensorflow/tfjs-node';

const CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];

let model = null;

async function loadModel() {
  if (!model) {
    model = await tf.loadLayersModel('file://public/tfjs_model/model.json');
  }
  return model;
}

export default async function handler(req, res) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  // Only accept POST requests
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed. Use POST.' });
    return;
  }

  try {
    // Parse form data to get uploaded file
    const form = formidable({ multiples: false, maxFileSize: 10 * 1024 * 1024 });
    const [fields, files] = await form.parse(req);

    const file = files.image?.[0] || null;
    if (!file) {
      return res.status(400).json({ error: 'No image file provided' });
    }

    // Process image with Sharp
    let imageBuffer = fs.readFileSync(file.filepath);

    // Resize image to model input size (224x224)
    const processedBuffer = await sharp(imageBuffer)
      .resize(224, 224, { fit: 'fill' })
      .removeAlpha()
      .jpeg({ quality: 90 })
      .toBuffer();

    // Load model if not already loaded
    const loadedModel = await loadModel();

    // Preprocess for model: normalize to [0,1]
    let imageTensor = tf.node.decodeImage(processedBuffer, 3);
    imageTensor = tf.image.resizeBilinear(imageTensor, [224, 224]);
    imageTensor = imageTensor.div(255.0);
    imageTensor = imageTensor.expandDims(0); // Add batch dimension

    // Run prediction
    const predictions = loadedModel.predict(imageTensor);
    const predictionArray = await predictions.array();
    const probabilities = predictionArray[0];

    // Get the predicted class
    const predictedIndex = probabilities.indexOf(Math.max(...probabilities));
    const predictedClass = CIFAR10_CLASSES[predictedIndex];
    const confidence = probabilities[predictedIndex];

    // Clean up tensors
    imageTensor.dispose();
    predictions.dispose();

    // Return processed image and prediction
    const base64Image = processedBuffer.toString('base64');
    const imageUrl = `data:image/jpeg;base64,${base64Image}`;

    const response = {
      success: true,
      imageUrl: imageUrl,
      prediction: {
        class: predictedClass,
        confidence: confidence,
        probabilities: probabilities.map((prob, idx) => ({
          class: CIFAR10_CLASSES[idx],
          probability: prob
        }))
      },
      modelInfo: {
        type: 'CIFAR-10 Transfer Learning Model',
        dataset: 'CIFAR-10',
        inputShape: [224, 224, 3],
        classes: 10,
        classNames: CIFAR10_CLASSES,
        status: 'Server-side classification complete'
      }
    };

    res.status(200).json(response);

  } catch (error) {
    console.error('Classification error:', error);
    res.status(500).json({
      success: false,
      error: 'Classification failed',
      details: error.message
    });
  }
}

export const config = {
  api: {
    bodyParser: false,
  },
};
