#!/usr/bin/env python3

print("Python is working!")

# Test importing tensorflow
try:
    import tensorflow as tf
    print("✅ TensorFlow imported successfully!")
    print(f"Version: {tf.__version__}")
except ImportError as e:
    print(f"❌ Failed to import TensorFlow: {e}")

# Check current directory
import os
print(f"Current directory: {os.getcwd()}")

# List available model files
model_files = [
    'cifar10_high_accuracy_model.h5',
    'best_cifar10_model.keras',
    '../../transfer_model.h5'
]

print("Looking for model files:")
for model_file in model_files:
    if os.path.exists(model_file):
        print(f"✅ Found: {model_file}")
    else:
        print(f"❌ Not found: {model_file}")
