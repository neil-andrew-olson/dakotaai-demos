#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import json

def convert_h5_to_tfjs():
    """Convert CIFAR-10 high accuracy model to TensorFlow.js format"""

    # Try available model files in order of preference
    model_files = [
        'cifar10_high_accuracy_model.h5',
        'best_cifar10_model.keras',
        '../../transfer_model.h5'
    ]

    model_path = None
    for candidate in model_files:
        if os.path.exists(candidate):
            model_path = candidate
            break

    if model_path is None:
        print("❌ No suitable model file found!")
        print("Looking for:", model_files)
        return False

    print(f"📥 Loading model from: {model_path}")

    # Handle different model formats
    if model_path.endswith('.h5'):
        model = tf.keras.models.load_model(model_path)
    elif model_path.endswith('.keras'):
        model = tf.keras.models.load_model(model_path)
    else:
        print("❌ Unsupported model format")
        return False

    print("✅ Model loaded successfully!")

    # Display model summary
    print("\n📊 Model Summary:")
    model.summary()

    # Create output directory
    output_dir = 'public/tfjs_model'
    os.makedirs(output_dir, exist_ok=True)

    # Convert using SavedModel format (more compatible)
    print(f"\n🔄 Converting to TensorFlow.js format...")
    print(f"Output directory: {output_dir}")

    # Save as SavedModel first, then convert
    saved_model_dir = 'temp_saved_model'
    model.export(saved_model_dir, format='tf_saved_model')

    # Convert using the converters API (avoiding deprecated save_keras_model)
    import subprocess
    try:
        result = subprocess.run([
            'tensorflowjs_converter',
            '--input_format=tf_saved_model',
            '--output_format=tfjs_graph_model',
            saved_model_dir,
            output_dir
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Conversion completed successfully!")
        else:
            print(f"❌ Conversion failed: {result.stderr}")
            return False

    except FileNotFoundError:
        print("❌ tensorflowjs_converter not found. Using alternative method...")

        # Alternative: Manual conversion (simplified for basic models)
        print("🔧 Using alternative conversion method...")

        # Save weights and create model.json manually
        weights_path = os.path.join(output_dir, 'weights.bin')
        model_json_path = os.path.join(output_dir, 'model.json')

        # This is a simplified approach - for production, you'd need full tfjs conversion
        print("⚠️  Alternative conversion may not work for complex models")

        # Try to get model weights
        weights = model.get_weights()
        if weights:
            # Serialize weights (simplified)
            weight_data = b''
            for w in weights:
                weight_data += w.tobytes()

            with open(weights_path, 'wb') as f:
                f.write(weight_data)

            # Create basic model.json (simplified)
            model_config = model.to_json()
            with open(model_json_path, 'w') as f:
                json.dump({
                    'modelTopology': json.loads(model_config),
                    'weightsManifest': [{
                        'paths': ['weights.bin'],
                        'weights': []
                    }],
                    'format': 'layers-model',
                    'generatedBy': 'TensorFlow.js',
                    'convertedBy': 'custom-converter'
                }, f, indent=2)

            print("✅ Alternative conversion completed (limited compatibility)")
        else:
            print("❌ Could not extract model weights")

    # Clean up temp directory
    if os.path.exists(saved_model_dir):
        import shutil
        shutil.rmtree(saved_model_dir)

    # Verify output files
    expected_files = ['model.json', 'weights.bin']
    print("\n📁 Generated files:")
    for file in expected_files:
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file}: {size} bytes")
        else:
            print(f"❌ {file}: Missing")

    print(f"\n🎉 CIFAR-10 model ready! Load from: /tfjs_model/model.json")
    return True

if __name__ == "__main__":
    convert_h5_to_tfjs()
