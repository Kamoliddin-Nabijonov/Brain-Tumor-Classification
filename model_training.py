import logging
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import (Conv2D, MaxPool2D, BatchNormalization, Dropout,
                          Dense, InputLayer, Flatten)
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu():
    """Check if GPU is available."""
    gpu_available = tf.config.list_physical_devices('GPU')
    if gpu_available:
        logger.info("GPU is available!")
        for gpu in gpu_available:
            logger.info(f"Device: {gpu}")
    else:
        logger.warning("No GPU detected.")

def prepare_dataset(data_dir, seed):
    """Prepare training and validation datasets."""
    image_data_generator = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2,
        zoom_range=(0.99, 0.99)
    )

    train = image_data_generator.flow_from_directory(
        data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary",
        color_mode="rgb",
        shuffle=True,
        seed=seed,
        subset="training"
    )

    validation = image_data_generator.flow_from_directory(
        data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary",
        color_mode="rgb",
        shuffle=True,
        seed=seed,
        subset="validation"
    )

    return train, validation

def build_model(learning_rate=0.001, dropout_rate=0.3, size_inner=128):
    """Build a Sequential model for binary classification."""
    model = keras.Sequential([
        InputLayer(input_shape=(150, 150, 3)),
        Conv2D(filters=32, kernel_size=3, activation="relu", padding="same"),
        MaxPool2D(),
        Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"),
        MaxPool2D(),
        Flatten(),
        Dense(size_inner, activation='relu'),
        BatchNormalization(),
        Dropout(rate=dropout_rate),
        Dense(size_inner // 2, activation="relu"),
        BatchNormalization(),
        Dropout(rate=dropout_rate),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=BinaryCrossentropy(),
        metrics=['accuracy']
    )

    return model

def main():
    """Main function to run the script."""
    SEED = 42
    DATA_DIR = "./Dataset/Images/"

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        './Trained_Models/Sequential_{epoch:02d}_{val_accuracy:.3f}.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )

    check_gpu()
    
    logger.info("Preparing datasets...")
    train_data, validation_data = prepare_dataset(DATA_DIR, SEED)

    logger.info("Building the model...")
    model = build_model(learning_rate=0.0001, dropout_rate=0.2, size_inner=128)

    logger.info("Starting training...")
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=50,
        callbacks=[checkpoint_callback],
        verbose=1
    )

    logger.info("Training complete.")

if __name__ == "__main__":
    main()
