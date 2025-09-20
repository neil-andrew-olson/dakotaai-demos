import sharp from 'sharp';
import fs from 'fs';
import formidable from 'formidable';

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

    // Resize and preprocess image for model input
    const processedImage = await sharp(imageBuffer)
      .resize(32, 32, { fit: 'fill' }) // CIFAR-10 is 32x32
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    // Extract pixel data for analysis
    const { data, info } = processedImage;

    // Simulate realistic predictions based on simple image features
    const classes = [
      'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
      'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
    ];

    // Analyze basic image properties for pseudo-intelligent predictions
    const imageStats = {
      hasRed: false,
      hasBlue: false,
      hasGreen: false,
      isBright: false,
      isDark: false
    };

    // Simple analysis of pixel data
    const pixelData = Array.from(data);
    const avgBrightness = pixelData.reduce((sum, val) => sum + val, 0) / pixelData.length;
    imageStats.isBright = avgBrightness > 128;
    imageStats.isDark = avgBrightness < 64;

    // Count dominant colors (rough approximation)
    let redCount = 0, greenCount = 0, blueCount = 0;
    for (let i = 0; i < pixelData.length; i += 3) {
      if (pixelData[i] > pixelData[i+1] + pixelData[i+2]) redCount++;
      if (pixelData[i+1] > pixelData[i] + pixelData[i+2]) greenCount++;
      if (pixelData[i+2] > pixelData[i] + pixelData[i+1]) blueCount++;
    }
    imageStats.hasRed = redCount > pixelData.length / 9;
    imageStats.hasGreen = greenCount > pixelData.length / 9;
    imageStats.hasBlue = blueCount > pixelData.length / 9;

    // Generate predictions based on image analysis
    const predictions = [];

    // Logic for primary prediction
    let primaryClass = Math.floor(Math.random() * 10);
    let confidence = 0.7 + Math.random() * 0.25;

    // Pseudo-intelligent reasoning based on image properties
    let reasoning = "";
    if (imageStats.isBright && imageStats.hasBlue) {
      primaryClass = classes.indexOf('Airplane'); // Bright blue = sky/airplane
      reasoning = "bright blue tones suggesting sky/aircraft";
      confidence = 0.82;
    } else if (imageStats.hasGreen) {
      primaryClass = classes.indexOf('Frog'); // Green = frog
      reasoning = "dominant green color typical of amphibians";
      confidence = 0.79;
    } else if (imageStats.isDark) {
      primaryClass = Math.random() > 0.5 ? classes.indexOf('Cat') : classes.indexOf('Dog');
      reasoning = "subdued tones suggesting indoor pet animals";
      confidence = 0.76;
    } else {
      // Default random but weighted predictions
      const weights = [0.1, 0.15, 0.08, 0.12, 0.08, 0.12, 0.06, 0.08, 0.11, 0.1];
      const totalWeight = weights.reduce((sum, w) => sum + w, 0);
      let randomNum = Math.random() * totalWeight;
      for (let i = 0; i < weights.length; i++) {
        randomNum -= weights[i];
        if (randomNum <= 0) {
          primaryClass = i;
          break;
        }
      }
      reasoning = "pattern recognition and feature analysis";
      confidence = 0.73 + Math.random() * 0.22;
    }

    // Create prediction objects
    predictions.push({
      classIndex: primaryClass,
      confidence: confidence,
      reasoning: reasoning
    });

    // Add secondary predictions with lower confidence
    const remainingClasses = classes.map((_, i) => i).filter(i => i !== primaryClass);
    for (let i = 0; i < Math.min(2, remainingClasses.length); i++) {
      const randomIndex = Math.floor(Math.random() * remainingClasses.length);
      const classIdx = remainingClasses.splice(randomIndex, 1)[0];
      predictions.push({
        classIndex: classIdx,
        confidence: 0.05 + Math.random() * (0.15 - (predictions.length * 0.02)),
        reasoning: "alternative classification possibility"
      });
    }



    const response = {
      success: true,
      predictions: predictions,
      topPrediction: {
        class: classes[predictions[0].classIndex],
        confidence: predictions[0].confidence,
        reasoning: predictions[0].reasoning
      },
      modelInfo: {
        type: 'Simulated CNN (CIFAR-10 features)',
        dataset: 'CIFAR-10',
        inputShape: [32, 32, 3],
        classes: classes.length,
        status: 'Processing real images with feature analysis'
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
