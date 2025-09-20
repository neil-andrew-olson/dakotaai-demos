#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import numpy as np
import os
import matplotlib.pyplot as plt
import math

def squeeze_excite_block(input_tensor, ratio=16):
    """Squeeze and Excitation block for channel attention"""
    filters = input_tensor.shape[-1]

    # Squeeze
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape((1, 1, filters))(se)

    # Excitation
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)

    # Scale
    return layers.Multiply()([input_tensor, se])

def residual_block(x, filters, kernel_size=(3, 3), strides=1, use_se=True):
    """Residual block with optional squeeze-excitation"""
    shortcut = x

    # Main path
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same', kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)

    # Squeeze-Excitation
    if use_se:
        x = squeeze_excite_block(x)

    # Shortcut connection
    if strides > 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=strides, padding='valid', kernel_regularizer=l2(0.0001))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    return x

def create_modern_cnn(input_shape=(32, 32, 3), num_classes=10):
    """Create a modern CNN architecture for CIFAR-10 that can achieve 95%+ accuracy """

    inputs = layers.Input(shape=input_shape)

    # Initial conv
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # First residual block (32 filters)
    x = residual_block(x, 32, use_se=True)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Second residual block (64 filters)
    x = residual_block(x, 64, use_se=True)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Third residual block (128 filters)
    x = residual_block(x, 128, use_se=True)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Fourth residual block (256 filters)
    x = residual_block(x, 256, use_se=True)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers
    x = layers.Dense(1024, kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def create_data_augmentation():
    """Create data augmentation pipeline"""

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.15,
        shear_range=0.1,
        fill_mode='nearest'
    )

    return datagen

def create_mixup_generator(datagen, alpha=0.2):
    """Create mixup data generator"""
    def mixup_generator():
        while True:
            x_batch, y_batch = next(datagen)
            batch_size = x_batch.shape[0]

            # Generate lambda values
            lam = np.random.beta(alpha, alpha, batch_size)

            # Shuffle batch
            indices = np.random.permutation(batch_size)
            x_shuffled = x_batch[indices]
            y_shuffled = y_batch[indices]

            # Mix images and labels
            x_mixed = lam.reshape(batch_size, 1, 1, 1) * x_batch + (1 - lam).reshape(batch_size, 1, 1, 1) * x_shuffled
            y_mixed = lam.reshape(batch_size, 1) * y_batch + (1 - lam).reshape(batch_size, 1) * y_shuffled

            yield x_mixed, y_mixed

    return mixup_generator

def create_enhanced_cnn(input_shape=(32, 32, 3), num_classes=10):
    """Enhanced CNN with progressive architecture for higher accuracy"""

    inputs = layers.Input(shape=input_shape)

    # Initial conv with larger filters
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # First residual block (64 filters)
    x = residual_block(x, 64, use_se=True)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Second residual block (128 filters)
    x = residual_block(x, 128, use_se=True)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Third residual block (256 filters)
    x = residual_block(x, 256, use_se=True)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.35)(x)

    # Fourth residual block (512 filters)
    x = residual_block(x, 512, use_se=True)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers with more regularization
    x = layers.Dense(512, kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def load_and_preprocess_cifar10():
    """Load CIFAR-10 dataset and preprocess for training"""

    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    return (x_train, y_train), (x_test, y_test)

def plot_training_history(history):
    """Plot training accuracy and loss curves"""

    plt.figure(figsize=(12, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('cifar10_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def warmup_cosine_decay_scheduler(epoch, lr):
    """Learning rate scheduler with warmup and cosine decay"""
    warmup_epochs = 10
    max_lr = 0.1
    min_lr = 0.001
    total_epochs = 200

    if epoch < warmup_epochs:
        return max_lr * ((epoch + 1) / warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def train_high_accuracy_model(use_enhanced=True):
    """Train a high-accuracy CIFAR-10 model with enhancements"""

    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_cifar10()

    # Create enhanced model
    if use_enhanced:
        print("\n🏗️ Building enhanced CNN architecture...")
        model = create_enhanced_cnn()
    else:
        print("\n🏗️ Building modern CNN architecture...")
        model = create_modern_cnn()

    model.summary()

    # Compile model with enhanced learning rate schedule
    initial_lr = 0.1
    epochs = 200

    # Create data augmentation
    datagen = create_data_augmentation()
    datagen.fit(x_train)

    # Optional: Enable mixup for additional regularization
    use_mixup = False
    if use_mixup:
        mixup_gen = create_mixup_generator(datagen.flow(x_train, y_train, batch_size=128), alpha=0.2)
        train_generator = mixup_gen
        steps_per_epoch = len(x_train) // 128
    else:
        train_generator = datagen.flow(x_train, y_train, batch_size=128)
        steps_per_epoch = len(x_train) // 128

    # Learning rate scheduler with warmup
    lr_callback = callbacks.LearningRateScheduler(warmup_cosine_decay_scheduler)

    optimizer = optimizers.SGD(learning_rate=initial_lr, momentum=0.9, nesterov=True)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks for better training
    callbacks_list = [
        lr_callback,
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            'best_cifar10_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=15,
            min_lr=0.0001,
            verbose=1
        )
    ]

    # Train the model
    print("\n🎯 Starting enhanced training for high accuracy...")
    print("Target: 95%+ validation accuracy on CIFAR-10")

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks_list,
        verbose=1
    )

    # Evaluate final model
    print("\n📊 Final Evaluation:")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(".4f")

    # Save the model for conversion
    print("\n💾 Saving model for TensorFlow.js conversion...")
    model.save('cifar10_high_accuracy_model.keras')
    model.save('cifar10_high_accuracy_model.h5')

    # Plot training history
    plot_training_history(history)

    print("\n✅ Training completed!")
    print(".4f")
    print("Model saved as:")
    print("  - cifar10_high_accuracy_model.keras (Keras format)")
    print("  - cifar10_high_accuracy_model.h5 (H5 format)")

    return model, history

def evaluate_model_comprehensive(model, x_test, y_test):
    """Comprehensive evaluation of the trained model"""

    print("\n🔬 Comprehensive Model Evaluation:")

    # Get predictions
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    # Calculate per-class accuracy
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    print("Per-class accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = (true_classes == i)
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(predicted_classes[class_mask] == true_classes[class_mask])
            print(".4f")

    # Confusion matrix (simplified)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    print(f"\nConfusion matrix shape: {cm.shape}")
    print("Most confused pairs (top 20):")
    # Find most confused pairs
    confusion_list = []
    for i in range(10):
        for j in range(10):
            if i != j:
                confusion_list.append((cm[i,j], class_names[i], class_names[j]))
    confusion_list.sort(reverse=True, key=lambda x: x[0])
    for count, class1, class2 in confusion_list[:20]:
        if count > 0:
            print(f"  {class1} ↔ {class2}: {count} times")
    print(f"\nGoal: Keep most confused pairs under 50 instances. Current max: {confusion_list[0][0] if confusion_list else 0}")

    return predictions, predicted_classes, true_classes

def evaluate_saved_model(model_path='best_cifar10_model.keras'):
    """Evaluate a saved model"""
    print(f"Loading model from {model_path}...")
    print(f"Current working directory: {os.getcwd()}")

    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully!")
        else:
            print(f"Model file not found: {model_path}")
            return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Load test data
    (_, _), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0
    y_test = to_categorical(y_test, 10)

    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(".4f")

    # Comprehensive evaluation
    evaluate_model_comprehensive(model, x_test, y_test)

    return model

if __name__ == "__main__":
    print("🚀 CIFAR-10 High-Accuracy Model Training")
    print("=" * 50)

    # Set memory growth for GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("✅ GPU memory growth enabled")
    else:
        print("⚠️ No GPU detected, training on CPU")

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
        # Evaluate saved model
        if len(sys.argv) > 2:
            model_path = sys.argv[2]
        else:
            model_path = 'best_cifar10_model.keras'
        evaluate_saved_model(model_path)
    else:
        # Train new model
        use_enhanced = True  # Enable enhanced architecture
        model, history = train_high_accuracy_model(use_enhanced=use_enhanced)

        # Load test data for comprehensive evaluation
        (_, _), (x_test, y_test) = cifar10.load_data()
        x_test = x_test.astype('float32') / 255.0
        y_test = to_categorical(y_test, 10)

        # Comprehensive evaluation
        evaluate_model_comprehensive(model, x_test, y_test)

        print("\n🎉 CIFAR-10 High-Accuracy Training Complete!")
        print("Next step: Convert the .h5 model to TensorFlow.js format")
        print("Run: python convert_model.py (update path to new model)")
