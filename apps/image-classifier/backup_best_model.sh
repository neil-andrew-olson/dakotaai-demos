#!/bin/bash

echo "📋 Creating backup of best CIFAR-10 model..."

# Create backup directory
mkdir -p model_backup

# Backup the best model
if [ -f "best_cifar10_model.keras" ]; then
    cp best_cifar10_model.keras model_backup/
    echo "✅ Backed up: best_cifar10_model.keras"
fi

if [ -f "cifar10_high_accuracy_model.keras" ]; then
    cp cifar10_high_accuracy_model.keras model_backup/
    echo "✅ Backed up: cifar10_high_accuracy_model.keras"
fi

echo "📂 Models backed up to: model_backup/"
echo ""
echo "🗑️  Ready to clean large files..."
echo "This will keep your working MobileNet classifier"
echo "and remove only the oversized model files"
