import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

def load_data(data_dir, img_size=(250, 250)):
    """Loads images from 'Testing' and 'Training and Validation' subdirectories."""
    images, labels = [], []
    
    def load_specific(folder, label):
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} not found.")
            return
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(os.path.join(folder, fname))
                if img is not None:
                    img = cv2.resize(img, img_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(label)
                    
    # Fire label: 1, No Fire label: 0
    load_specific(os.path.join(data_dir, 'Testing', 'fire'), 1)
    load_specific(os.path.join(data_dir, 'Testing', 'nofire'), 0)
    load_specific(os.path.join(data_dir, 'Training and Validation', 'fire'), 1)
    load_specific(os.path.join(data_dir, 'Training and Validation', 'nofire'), 0)
    
    return images, labels

def augment_and_normalize(images, labels):
    """Applies horizontal flip augmentation and scale to [0, 1]."""
    aug_images, aug_labels = [], []
    for img, lbl in zip(images, labels):
        # Original
        aug_images.append(img)
        aug_labels.append(lbl)
        # Flipped
        aug_images.append(cv2.flip(img, 1))
        aug_labels.append(lbl)

    X_aug = np.array(aug_images, dtype=np.float32)
    y_aug = np.array(aug_labels)
    
    # Min-Max Normalization
    x_min, x_max = 0.0, 255.0
    X_norm = (X_aug - x_min) / (x_max - x_min)
    
    return X_norm, y_aug

def build_model(input_shape=(250, 250, 3)):
    """Constructs the CNN model architecture."""
    model = models.Sequential([
        # Block 1 - Standard
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=1, input_shape=input_shape),
        layers.MaxPooling2D((3, 3)),

        # Block 2 - Separable 1
        layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((3, 3)),
        layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((3, 3)),

        # Block 3 - Separable 2
        layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
        layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
        layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        # Fully Connected Layers
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.30),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.30),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.20),

        layers.Dense(2, activation='softmax')
    ])
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=150, batch_size=32, save_path='best_model.keras'):
    """Compiles and trains the model, saving the best version."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    callbacks = [
        ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    end_time = time.time()
    print(f"Training Time: {end_time - start_time:.2f} seconds")
    
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the test set, computing QWK and Confusion Matrix."""
    y_pred_proba = model.predict(X_test)
    y_pred_cls = np.argmax(y_pred_proba, axis=1)
    y_true_cls = np.argmax(y_test, axis=1)

    kappa = cohen_kappa_score(y_true_cls, y_pred_cls, weights='quadratic')
    print(f"QWK (Cohen's Kappa): {kappa*100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_true_cls, y_pred_cls))

    cm = confusion_matrix(y_true_cls, y_pred_cls)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Fire', 'Fire'], yticklabels=['No Fire', 'Fire'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def perform_inference(model_path, sample_img):
    """Loads a pre-trained model and predicts the class of a single image."""
    try:
        model = load_model(model_path)
        pred = model.predict(sample_img)
        status = "FIRE DETECTED" if np.argmax(pred) == 1 else "Clear"
        print(f"Inference Result: {status} | Probabilities: {pred}")
    except Exception as e:
        print(f"Failed to perform inference: {e}")

def main():
    # 1. Configuration Check
    data_dir = 'mendeley_dataset'
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found.")
        return
        
    print("Loading data...")
    images, labels = load_data(data_dir)
    print(f"Initial dataset size: {len(images)} images")
    
    if not images:
        print("No images found. Exiting.")
        return

    # 2. Augmentation & Normalization
    print("\nAugmenting and normalizing data...")
    X_norm, y_aug = augment_and_normalize(images, labels)
    print(f"After horizontal flip: {len(X_norm)} images")
    print(f"Normalized pixel range: [{X_norm.min():.1f}, {X_norm.max():.1f}]")

    # 3. Splitting and Encoding
    y_cat = to_categorical(y_aug, num_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y_cat,
        test_size=0.20,
        random_state=42,
        stratify=y_cat
    )
    print(f"Train size: {X_train.shape[0]}")
    print(f"Test size: {X_test.shape[0]}")

    # 4. Model Creation
    print("\nBuilding model...")
    model = build_model()
    model.summary()

    # 5. Training
    model_save_path = 'best_model.keras'
    print("\nStarting training...")
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=150, batch_size=32, save_path=model_save_path)

    # 6. Evaluation
    print("\nEvaluating best model...")
    best_model = load_model(model_save_path)
    evaluate_model(best_model, X_test, y_test)

    # 7. Inference Test
    print("\nRunning test inference...")
    perform_inference(model_save_path, X_test[0:1])

def run(dataset_path):
    """Standard pipeline interface. dataset_path: Mendeley-style dir with Testing/ and Training and Validation/ sub-folders."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import load_model as _load_keras_model

    if not os.path.exists(dataset_path):
        return {"model_name": "UAV-DNN", "error": f"Dataset not found: {dataset_path}", "metrics": None}

    images, labels = load_data(dataset_path)
    if not images:
        return {"model_name": "UAV-DNN", "error": "No images found", "metrics": None}

    X_norm, y_aug = augment_and_normalize(images, labels)
    y_cat = to_categorical(y_aug, num_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y_cat, test_size=0.20, random_state=42, stratify=y_cat
    )

    model = build_model()
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uav_dnn_best.keras')
    train_model(model, X_train, y_train, X_test, y_test, epochs=150, batch_size=32, save_path=save_path)

    best_model = _load_keras_model(save_path)
    y_pred_proba = best_model.predict(X_test)
    y_pred_cls   = np.argmax(y_pred_proba, axis=1)
    y_true_cls   = np.argmax(y_test, axis=1)

    tp = np.sum((y_pred_cls == 1) & (y_true_cls == 1))
    tn = np.sum((y_pred_cls == 0) & (y_true_cls == 0))
    fp = np.sum((y_pred_cls == 1) & (y_true_cls == 0))
    fn = np.sum((y_pred_cls == 0) & (y_true_cls == 1))
    total = tp + tn + fp + fn

    accuracy  = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp)   if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn)   if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    auc_val   = roc_auc_score(y_true_cls, y_pred_proba[:, 1])
    aupr_val  = average_precision_score(y_true_cls, y_pred_proba[:, 1])

    return {
        "model_name": "UAV-DNN",
        "metrics": {
            "accuracy":  float(accuracy),
            "precision": float(precision),
            "recall":    float(recall),
            "f1":        float(f1),
            "auc":       float(auc_val),
            "aupr":      float(aupr_val),
        },
    }


if __name__ == '__main__':
    main()
