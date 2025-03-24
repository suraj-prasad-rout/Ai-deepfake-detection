import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from google.colab import drive

# ✅ Enable Mixed Precision & XLA Optimization
tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

# ✅ Mount Google Drive
drive.mount('/content/drive')

# ✅ Define Paths
frames_folder = "/content/drive/MyDrive/frames"
model_save_path = "/content/drive/MyDrive/deepfake_best_model.h5"

print(f"✅ Google Drive mounted! Dataset path: {frames_folder}")


def focal_loss(alpha=0.25, gamma=2.0):
    """Focal Loss to handle class imbalance"""
    def loss_fn(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        loss = alpha * (1 - p_t) ** gamma * bce
        return tf.reduce_mean(loss)
    return loss_fn


def load_data_with_generator(frames_folder, img_size=(160, 160), batch_size=12):
    """Load and preprocess data with optimized augmentation."""
    print("\n🔄 Loading Data with Augmentation...")

    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=0.2,
        rotation_range=15,  # Less distortion
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],  # Less brightness variation
        shear_range=0.1,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        frames_folder,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        frames_folder,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    print(
        f"✅ Train Samples: {train_generator.samples}, Validation Samples: {validation_generator.samples}")
    print(
        f"✅ Class distribution - Real: {np.bincount(train_generator.classes)[0]}, Fake: {np.bincount(train_generator.classes)[1]}")

    return train_generator, validation_generator


def create_model(img_size=(160, 160)):
    """Create an optimized CNN model for deepfake detection."""
    print("\n🔨 Creating Optimized CNN Model...")

    base_model = EfficientNetB7(
        weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

    # ✅ Increase trainable layers from 25% → 50% for better learning
    trainable_layers = int(len(base_model.layers) * 0.5)
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    for layer in base_model.layers[-trainable_layers:]:
        layer.trainable = True

    print(
        f"✅ Fine-tuning the last {trainable_layers} layers out of {len(base_model.layers)} total layers.")

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),

        layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(
            0.0005)),  # Reduced L2 Regularization
        layers.LeakyReLU(alpha=0.1),
        layers.BatchNormalization(),
        layers.Dropout(0.3),  # Reduced dropout for better feature retention

        layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        layers.LeakyReLU(alpha=0.1),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        layers.LeakyReLU(alpha=0.1),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=2e-4, weight_decay=5e-6)  # Slightly lower weight decay

    model.compile(
        optimizer=optimizer,
        loss=focal_loss(),
        metrics=['accuracy', tf.keras.metrics.AUC(
        ), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    print("✅ Model Architecture:")
    model.summary()

    return model


def train_model(train_generator, validation_generator):
    """Train the deepfake detection model."""
    print("\n🚀 Starting Model Training...")

    model = create_model(img_size=(160, 160))

    class_counts = np.bincount(train_generator.classes)
    class_weights = {
        0: class_counts[1] / class_counts[0], 1: class_counts[0] / class_counts[1]}
    print(f"📊 Class Weights Applied: {class_weights}")

    early_stop = EarlyStopping(
        monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1, mode='max')
    lr_schedule = ReduceLROnPlateau(
        monitor='val_loss', factor=0.3, patience=5, min_lr=5e-8, verbose=1)
    checkpoint = ModelCheckpoint(
        model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')

    start_time = time.time()
    history = model.fit(
        train_generator,
        epochs=60,
        validation_data=validation_generator,
        callbacks=[early_stop, lr_schedule, checkpoint],
        class_weight=class_weights
    )

    end_time = time.time()
    print(f"✅ Training completed in {(end_time - start_time)/60:.2f} minutes.")
    return model, history


def evaluate_model(model, validation_generator):
    """Evaluate the trained model."""
    print("\n📊 Evaluating Model...")
    results = model.evaluate(validation_generator, verbose=1)
    print(f"\n✅ Final Accuracy: {results[1]*100:.2f}%")
    return results


# ✅ Load Data
train_generator, validation_generator = load_data_with_generator(
    frames_folder, img_size=(160, 160), batch_size=12)

# ✅ Train Model
model, history = train_model(train_generator, validation_generator)

# ✅ Evaluate Model
evaluate_model(model, validation_generator)

print(f"\n✅ Model saved to: {model_save_path}")
print("💯 Process completed successfully!")
