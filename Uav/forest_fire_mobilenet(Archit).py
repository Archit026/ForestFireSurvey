import os
import cv2
import numpy as np
import argparse

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model, load_model
except ImportError:
    print("Warning: TensorFlow is not installed. Please install it using 'pip install tensorflow'")


def setup_directories(dataset_dir='./dataset', test_dir='./test_images'):
    """Create necessary directories if they don't exist."""
    print("Setting up directories...")
    os.makedirs(os.path.join(dataset_dir, 'fire'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'non_fire_images'), exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)


def prepare_data_generators(dataset_dir, batch_size=32, img_size=(224, 224)):
    """Prepare training and validation data generators with augmentation."""
    print("Preparing data generators...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    return train_generator, val_generator


def build_model(input_shape=(224, 224, 3)):
    """Build and compile the MobileNetV2 transfer learning model."""
    print("Building the model...")
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(model, train_generator, val_generator, epochs=30, save_path='./fire_model.h5'):
    """Train the model and save it."""
    print(f"\nStarting training for {epochs} epochs...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )
    
    model.save(save_path)
    print(f"Model saved to {save_path}\n")
    return history


def run_inference(model_path, test_dir, img_size=(224, 224)):
    """Run inference on a single test image from the test directory."""
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train the model first.")
        return

    print("Loading model for inference...")
    model = load_model(model_path)
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory '{test_dir}' not found.")
        return

    images = os.listdir(test_dir)
    if not images:
        print(f"No images found in test folder '{test_dir}'.")
        return

    # Use the first image for demonstration
    test_image_path = os.path.join(test_dir, images[0])
    print(f"Using image for prediction: {test_image_path}")

    img = cv2.imread(test_image_path)
    if img is None:
        print("Error: Could not read image.")
        return

    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = np.reshape(img, (1, img_size[0], img_size[1], 3))

    pred = model.predict(img)[0][0]
    print(f"Raw Prediction Value: {pred:.4f}")

    # The notebook mapped pred < 0.5 to FIRE based on the binary class index mapping
    if pred < 0.5:
        print("Prediction: FIRE 🔥")
    else:
        print("Prediction: NON FIRE 🌲")


def main():
    parser = argparse.ArgumentParser(description="MobileNetV2 Forest Fire Detection Pipeline")
    parser.add_argument('--dataset_dir', type=str, default='./dataset', help="Path to the dataset directory")
    parser.add_argument('--test_dir', type=str, default='./test_images', help="Path to the test images directory")
    parser.add_argument('--model_path', type=str, default='./fire_model.h5', help="Path to save/load the model")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=30, help="Number of training epochs")
    parser.add_argument('--mode', type=str, choices=['train', 'infer', 'all'], default='all', help="Execution mode")
    
    args = parser.parse_args()

    setup_directories(args.dataset_dir, args.test_dir)

    if args.mode in ['train', 'all']:
        train_gen, val_gen = prepare_data_generators(args.dataset_dir, args.batch_size)
        model = build_model()
        train_model(model, train_gen, val_gen, epochs=args.epochs, save_path=args.model_path)

    if args.mode in ['infer', 'all']:
        run_inference(args.model_path, args.test_dir)


def run(dataset_path):
    """Standard pipeline interface.
    dataset_path: root dir with 'fire/' and 'non_fire_images/' sub-folders.
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
    )

    if not os.path.exists(dataset_path):
        return {"model_name": "MobileNetV2-UAV", "error": f"Dataset not found: {dataset_path}", "metrics": None}

    try:
        train_gen, val_gen = prepare_data_generators(dataset_path)
        model = build_model()
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mobilenet_fire.h5')
        train_model(model, train_gen, val_gen, epochs=30, save_path=save_path)

        # Collect predictions on the full validation set
        val_gen.reset()
        y_score_raw = model.predict(val_gen, verbose=0).flatten()
        y_true_raw  = val_gen.classes

        # Determine which class index is 'fire'
        fire_idx = val_gen.class_indices.get('fire', 0)
        if fire_idx == 0:
            # class 0 = fire => low score means fire in binary sigmoid output
            y_true_bin   = (y_true_raw == 0).astype(int)
            y_score_fire = 1.0 - y_score_raw
        else:
            y_true_bin   = (y_true_raw == fire_idx).astype(int)
            y_score_fire = y_score_raw

        y_pred   = (y_score_fire >= 0.5).astype(int)
        has_both = len(np.unique(y_true_bin)) > 1

        return {
            "model_name": "MobileNetV2-UAV",
            "metrics": {
                "accuracy":  float(accuracy_score(y_true_bin, y_pred)),
                "precision": float(precision_score(y_true_bin, y_pred, zero_division=0)),
                "recall":    float(recall_score(y_true_bin, y_pred, zero_division=0)),
                "f1":        float(f1_score(y_true_bin, y_pred, zero_division=0)),
                "auc":       float(roc_auc_score(y_true_bin, y_score_fire))       if has_both else None,
                "aupr":      float(average_precision_score(y_true_bin, y_score_fire)) if has_both else None,
            },
        }
    except Exception as exc:
        return {"model_name": "MobileNetV2-UAV", "error": str(exc), "metrics": None}


if __name__ == "__main__":
    main()
