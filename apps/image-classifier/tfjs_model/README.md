# TensorFlow.js Model Setup for Image Classifier Demo

This directory should contain your TensorFlow.js converted model files from your transfer-image-classifier repository.

## 🔧 Manual Model Conversion (Fixed)

Due to version compatibility issues, use this step-by-step approach:

### Step 1: Navigate to Model Directory
```bash
cd "C:\Users\neil\Desktop\transfer-image-classifier"
```

### Step 2: Create Python Conversion Script
Create a file called `manual_convert.py`:

```python
import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs
import os

# Load your trained model
model_path = "models/transfer_model.h5"
model = keras.models.load_model(model_path)

print("Model loaded successfully!")
print("Model summary:")
model.summary()

# Check model compatibility with TensorFlow.js
def check_tfjs_compatibility(model):
    """Check if model is compatible with TensorFlow.js"""
    unsupported_layers = []

    for layer in model.layers:
        layer_type = type(layer).__name__
        # List of TensorFlow.js supported layer types
        supported_types = [
            'InputLayer', 'Dense', 'Conv2D', 'MaxPooling2D', 'AveragePooling2D',
            'GlobalAveragePooling2D', 'GlobalMaxPooling2D', 'BatchNormalization',
            'Activation', 'Dropout', 'Flatten', 'Reshape', 'Concatenate',
            'Add', 'Multiply', 'Rescaling', 'ZeroPadding2D'
        ]

        if layer_type not in supported_types:
            unsupported_layers.append(f"{layer_type}: {layer.name}")

    return unsupported_layers

incompatible_layers = check_tfjs_compatibility(model)
if incompatible_layers:
    print("⚠️  Warning: Found potentially incompatible layers:")
    for layer in incompatible_layers:
        print(f"  - {layer}")
    print("\nSome layers might need to be converted or removed for TensorFlow.js compatibility")
else:
    print("✅ All layers appear TensorFlow.js compatible")

# Convert the model
try:
    tfjs.converters.save_keras_model(model, "./tfjs_model_converted")
    print("✅ Model converted successfully to ./tfjs_model_converted/")
except Exception as e:
    print(f"❌ Conversion failed: {e}")
    print("\nTroubleshooting:")
    print("1. Try with TensorFlow 2.13-2.15 for better compatibility")
    print("2. Some advanced layers might need custom conversion")
    print("3. Check model architecture for unsupported operations")
```

### Step 3: Run Manual Conversion
```bash
python manual_convert.py
```

### Step 4: Copy Converted Model
```bash
# If conversion succeeds
copy "C:\Users\neil\Desktop\transfer-image-classifier\tfjs_model_converted\*.*" "C:\Users\neil\Desktop\AIprojects\dakotaai-site\demos\tfjs_model\"
```

### Step 3: Verify the Conversion

After conversion, you should have these files in the `demos/tfjs_model/` directory:
- `model.json` - Model architecture and configuration
- `group1-shard1of1.bin` - Model weights (may be multiple shards)

### Step 4: Test the Demo

Open `demos/image_classifier.html` in your browser. The demo will:
1. Load your converted TensorFlow.js model
2. Allow image uploads for real classification
3. Display actual predictions from your trained transfer learning model

## 📋 Model Specifications

Your model should be trained to recognize CIFAR-10 classes:
- Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

**Input Shape:** 224x224x3 (RGB images)
**Output:** 10 classes with softmax probabilities

## 🔧 Troubleshooting

### Model Fails to Load
- Check that all TensorFlow.js files are in `demos/tfjs_model/`
- Ensure `tensorflowjs_converter` is version compatible with your TensorFlow version
- Verify your model architecture is compatible with TensorFlow.js

### Low Accuracy
- Make sure your model was properly trained on CIFAR-10 data
- Check that the input preprocessing matches your training preprocessing
- Verify the model architecture uses supported TensorFlow.js operations

### Performance Issues
- Consider using smaller model versions for web deployment
- Check browser console for WebGL backend issues

## 📖 More Resources

- [TensorFlow.js Model Conversion](https://www.tensorflow.org/js/tutorials/conversion/import_keras)
- [Supported Operations](https://github.com/tensorflow/tfjs/blob/master/tfjs-core/src/ops/ops.ts)
- [Model Optimization](https://www.tensorflow.org/js/guide/model_management)
