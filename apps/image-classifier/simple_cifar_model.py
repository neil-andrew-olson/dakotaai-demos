#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import json
import shutil

def create_simple_cifar_model():
    """Create a simple TF.js-compatible CIFAR-10 model from scratch"""

    print("🏗️ Creating simple TF.js-compatible CIFAR-10 model...")

    # Create a Sequential model with clean architecture
    model = models.Sequential([
        # Input layer (required by TF.js)
        layers.Input(shape=(32, 32, 3)),

        # Simple conv layers (no complex nested structures)
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    print("✅ Simple model architecture created")

    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("✅ Model compiled")

    # Try to train on a small subset first
    print("\n📚 Loading CIFAR-10 data...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print(f"Training data shape: {x_train.shape}")

    # Train for just a few epochs to get a working model
    print("\n🎯 Training simple model (5 epochs)...")
    model.fit(
        x_train[:10000], y_train[:10000],  # Use subset for quick training
        epochs=5,
        batch_size=128,
        validation_data=(x_test[:2000], y_test[:2000]),
        verbose=1
    )

    # Evaluate
    print("\n📊 Evaluating...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(".4f")

    # Save the model in separate formats
    print("\n💾 Saving model...")

    # Save as H5
    h5_path = 'simple_cifar10.h5'
    model.save(h5_path)
    print(f"✅ Saved as {h5_path}")

    # Convert to TF.js format
    print("\n🔄 Converting to TensorFlow.js format...")
    output_dir = 'public/simple_tfjs_model'

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Try using the TensorFlow.js converter
    try:
        import subprocess

        # Save as SavedModel first
        saved_model_dir = 'temp_simple_model'
        model.export(saved_model_dir, format='tf_saved_model')

        # Convert with tensorflowjs_converter
        result = subprocess.run([
            'tensorflowjs_converter',
            '--input_format=tf_saved_model',
            '--output_format=tfjs_layers_model',
            saved_model_dir,
            output_dir
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("✅ TF.js conversion successful!")
        else:
            print("❌ TF.js conversion failed, using manual method")
            manual_conversion(model, output_dir)

        # Clean up temp directory
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)

    except Exception as e:
        print(f"❌ Automatic conversion failed: {e}")
        manual_conversion(model, output_dir)

    # Verify output
    print(f"\n📁 Output directory: {output_dir}")
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        print(f"Files created: {files}")

        for file in files:
            file_path = os.path.join(output_dir, file)
            size = os.path.getsize(file_path)
            print("6s")

    print("
🎉 Simple CIFAR-10 model ready!")
    print("📂 Model location: /simple_tfjs_model/model.json"
    return True

def manual_conversion(model, output_dir):
    """Manual TF.js conversion for cases where converter fails"""
    print("🔧 Using manual TF.js conversion...")

    # Get model weights
    weights = model.get_weights()
    if not weights:
        print("❌ Could not extract model weights")
        return False

    # Save weights as binary file
    weight_data = b''
    for w in weights:
        weight_data += w.tobytes()

    weights_path = os.path.join(output_dir, 'weights.bin')
    with open(weights_path, 'wb') as f:
        f.write(weight_data)

    print("✅ Weights saved to weights.bin")

    # Create model.json with layer information
    model_json = {
        'modelTopology': {
            'keras_version': tf.keras.__version__,
            'backend': 'tensorflow',
            'model_config': {
                'class_name': 'Sequential',
                'config': {
                    'name': 'sequential',
                    'layers': [
                        {
                            'class_name': 'InputLayer',
                            'config': {
                                'batch_input_shape': [None, 32, 32, 3],
                                'dtype': 'float32',
                                'sparse': False,
                                'name': 'input_layer'
                            }
                        },
                        {
                            'class_name': 'Conv2D',
                            'config': {
                                'filters': 32,
                                'kernel_size': [3, 3],
                                'strides': [1, 1],
                                'padding': 'same',
                                'activation': 'relu',
                                'name': 'conv2d'
                            }
                        },
                        {
                            'class_name': 'BatchNormalization',
                            'config': {'name': 'batch_normalization'}
                        },
                        {
                            'class_name': 'MaxPooling2D',
                            'config': {
                                'pool_size': [2, 2],
                                'strides': [2, 2],
                                'padding': 'valid',
                                'name': 'max_pooling2d'
                            }
                        },
                        {
                            'class_name': 'Conv2D',
                            'config': {
                                'filters': 64,
                                'kernel_size': [3, 3],
                                'strides': [1, 1],
                                'padding': 'same',
                                'activation': 'relu',
                                'name': 'conv2d_1'
                            }
                        },
                        {
                            'class_name': 'BatchNormalization',
                            'config': {'name': 'batch_normalization_1'}
                        },
                        {
                            'class_name': 'MaxPooling2D',
                            'config': {
                                'pool_size': [2, 2],
                                'strides': [2, 2],
                                'padding': 'valid',
                                'name': 'max_pooling2d_1'
                            }
                        },
                        {
                            'class_name': 'Conv2D',
                            'config': {
                                'filters': 128,
                                'kernel_size': [3, 3],
                                'strides': [1, 1],
                                'padding': 'same',
                                'activation': 'relu',
                                'name': 'conv2d_2'
                            }
                        },
                        {
                            'class_name': 'BatchNormalization',
                            'config': {'name': 'batch_normalization_2'}
                        },
                        {
                            'class_name': 'MaxPooling2D',
                            'config': {
                                'pool_size': [2, 2],
                                'strides': [2, 2],
                                'padding': 'valid',
                                'name': 'max_pooling2d_2'
                            }
                        },
                        {
                            'class_name': 'Flatten',
                            'config': {'name': 'flatten'}
                        },
                        {
                            'class_name': 'Dense',
                            'config': {
                                'units': 256,
                                'activation': 'relu',
                                'name': 'dense'
                            }
                        },
                        {
                            'class_name': 'Dropout',
                            'config': {
                                'rate': 0.5,
                                'name': 'dropout'
                            }
                        },
                        {
                            'class_name': 'Dense',
                            'config': {
                                'units': 10,
                                'activation': 'softmax',
                                'name': 'dense_1'
                            }
                        }
                    ]
                }
            }
        },
        'weightsManifest': [{
            'paths': ['weights.bin'],
            'weights': []
        }],
        'format': 'layers-model',
        'generatedBy': 'TensorFlow.js',
        'convertedBy': 'simple-model-generator'
    }

    json_path = os.path.join(output_dir, 'model.json')
    with open(json_path, 'w') as f:
        json.dump(model_json, f, indent=2)

    print("✅ model.json created")
    return True

if __name__ == "__main__":
    print("🚀 Simple CIFAR-10 Model Creator for TF.js")
    print("=" * 50)

    success = create_simple_cifar_model()
    if success:
        print("\n📱 Ready to use with:")
        print("   - Frontend: /simple_tfjs_model/model.json")
        print("   - Will work in browser without InputLayer errors")
    else:
        print("\n❌ Simple model creation failed")
        print("   Consider manual model training")
