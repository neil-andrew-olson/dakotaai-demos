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

    // Convert processed image back to base64 for client-side AI processing
    const processedBuffer = await sharp(imageBuffer)
      .resize(224, 224, { fit: 'fill' }) // MobileNet standard size
      .jpeg({ quality: 90 })
      .toBuffer();

    const base64Image = processedBuffer.toString('base64');
    const imageUrl = `data:image/jpeg;base64,${base64Image}`;

    // Return processed image for client-side AI inference
    const response = {
      success: true,
      imageUrl: imageUrl,
      modelInfo: {
        type: 'MobileNet V2 (Real AI)',
        dataset: 'ImageNet',
        inputShape: [224, 224, 3],
        classes: 1000,
        status: 'Ready for client-side AI inference'
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
