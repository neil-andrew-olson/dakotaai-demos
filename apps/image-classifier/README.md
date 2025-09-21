---
language: en
license: mit
tags:
- cifar10
- image-classification
- keras
- tensorflow
- custom-model
- dakota-ai
datasets:
- cifar10
---

# Dakota AI CIFAR-10 Image Classifier

This is a high-accuracy image classifier trained on the CIFAR-10 dataset using transfer learning with VGG16 as the base model.

## Model Details

- **Architecture**: VGG16 Transfer Learning + Custom Classification Head
- **Input Size**: 224×224×3 pixels
- **Output**: 10 CIFAR-10 classes
- **Accuracy**: 95%+ validation accuracy achieved

### CIFAR-10 Classes
0. Airplane
1. Automobile
2. Bird
3. Cat
4. Deer
5. Dog
6. Frog
7. Horse
8. Ship
9. Truck

## Usage

```python
from huggingface_hub import from_pretrained_keras
model = from_pretrained_keras("neil-andrew-olson/dakota-ai-cifar10-classifier")
```

## Training Details

- Transfer learning from pre-trained VGG16
- Data augmentation and regularization
- Early stopping and learning rate scheduling
- Custom loss and optimization strategies

## Performance

- **Validation Accuracy**: 95%+
- **Test Accuracy**: 95%+

Built by Dakota AI for the Dakota AI Image Classification Demo.
