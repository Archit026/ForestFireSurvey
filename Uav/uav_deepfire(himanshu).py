import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def check_dataset_exists(data_dir: str):
    """Check if the dataset directory exists."""
    if not os.path.exists(data_dir):
        print(f"Dataset directory not found at {data_dir}!")
        return False
    return True


def load_and_preprocess_data(data_dir: str, img_size: tuple = (128, 128), test_size: float = 0.2):
    """Loads images, applies ROI cropping, resizes, and splits the dataset."""
    sub_dirs = ['Testing', 'Training and Validation']
    classes = {'fire': 1, 'nofire': 0}
    
    images, labels = [], []
    
    for sub in sub_dirs:
        for cls, label in classes.items():
            folder = os.path.join(data_dir, sub, cls)
            if not os.path.exists(folder):
                continue
                
            for fname in os.listdir(folder):
                path = os.path.join(folder, fname)
                img = cv2.imread(path)
                if img is None: 
                    continue
                
                # ROI Cropping
                h, w = img.shape[:2]
                margin_h, margin_w = int(h * 0.1), int(w * 0.1)
                img = img[margin_h:h-margin_h, margin_w:w-margin_w]
                
                # Resizing
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                images.append(img)
                labels.append(label)

    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels)
    
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


def train_ml_classifiers(X_train, y_train, X_test):
    """Train traditional ML classifiers and return their predictions."""
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    predictions = {}
    
    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_flat, y_train)
    predictions['KNN'] = knn.predict(X_test_flat)
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_flat, y_train)
    predictions['NB'] = nb.predict(X_test_flat)
    
    # SVM
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train_flat, y_train)
    predictions['SVM'] = svm.predict(X_test_flat)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_flat, y_train)
    predictions['RF'] = rf.predict(X_test_flat)
    
    # Logistic Regression
    lr = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    lr.fit(X_train_flat, y_train)
    predictions['LR'] = lr.predict(X_test_flat)
    
    return predictions


def build_and_train_vgg19(X_train, y_train, X_test, y_test, img_size=(128, 128)):
    """Build and train VGG19 Transfer Learning model."""
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.1,
        height_shift_range=0.1, horizontal_flip=True, zoom_range=0.1
    )

    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=64),
        epochs=50,
        validation_data=(X_test, y_test),
        verbose=1
    )

    model.save('deepfire_vgg19.keras')
    print('Model saved to deepfire_vgg19.keras')
    return model, history


def evaluate_model(name, y_true, y_pred):
    """Calculates evaluation metrics based on confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    er = (fp + fn) / (tp + tn + fp + fn)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    return {
        'Model': name, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'ER': er, 'F1': f1
    }


def generate_visualizations(y_test, predictions_dict, vgg_probs, history):
    """Generates ROC, PR, Learning curves and Confusion Matrices."""
    # Plot CMs
    for name, pred in predictions_dict.items():
        cm = confusion_matrix(y_test, pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, vgg_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # PR Curve
    precision_arr, recall_arr, _ = precision_recall_curve(y_test, vgg_probs)
    ap = average_precision_score(y_test, vgg_probs)
    plt.figure()
    plt.plot(recall_arr, precision_arr, label=f'PR (AP = {ap:.4f})')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    # Learning Curves
    if history:
        plt.figure()
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title('Model Accuracy')
        plt.legend()
        plt.show()


def simulate_uav_detection(model, image_path, threshold=0.5):
    """Simulates UAV real-time fire detection on a single image."""
    if not os.path.exists(image_path):
        return "File not found", 0.0
    
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = np.expand_dims(img, axis=0)
    
    prob = model.predict(img, verbose=0)[0][0]
    decision = 'FIRE DETECTED' if prob >= threshold else 'NO FIRE'
    
    print(f'Image: {image_path} | Prob: {prob:.4f} | Decision: {decision}')
    if decision == 'FIRE DETECTED':
        print('ACTION: Broadcast to nearby UAVs!')
        
    return decision, prob


def main():
    data_dir = '/kaggle/input/datasets/himankag8/mendeley-dataset'
    
    if not check_dataset_exists(data_dir):
        # We might want to fallback to the local mendeley_dataset in workspace if Kaggle path isn't found
        local_dir = 'mendeley_dataset'
        if os.path.exists(local_dir):
            print(f"Falling back to local dataset at {local_dir}")
            data_dir = local_dir
        else:
            return
            
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_dir)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    print("Training ML Classifiers...")
    ml_predictions = train_ml_classifiers(X_train, y_train, X_test)
    
    print("Building and training VGG19...")
    vgg_model, vgg_history = build_and_train_vgg19(X_train, y_train, X_test, y_test)
    
    print("Evaluating models...")
    # Predict with VGG19
    vgg_probs = vgg_model.predict(X_test).flatten()
    vgg_pred = (vgg_probs >= 0.5).astype(int)
    
    ml_predictions['VGG19-TL'] = vgg_pred
    
    # Generate unified results summary
    results = []
    for name, pred in ml_predictions.items():
        results.append(evaluate_model(name, y_test, pred))
        
    df_results = pd.DataFrame(results).round(4)
    print("\n--- Model Comparison Summary ---")
    print(df_results.to_string(index=False))
    df_results.to_csv('results_comparison_script.csv', index=False)
    
    print("\nGenerating visualizations...")
    generate_visualizations(y_test, ml_predictions, vgg_probs, vgg_history)
    
    # Run a test simulation on the first test image if debugging/testing locally is needed
    # (Leaving it commented for production execution purity)
    # print("\nSimulating UAV Detection...")
    # simulate_uav_detection(vgg_model, 'dummy_path.jpg')


def run(dataset_path):
    """Standard pipeline interface. Reports VGG19-TL as primary model.
    dataset_path: Mendeley-style dir with Testing/ and Training and Validation/ sub-folders.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    if not check_dataset_exists(dataset_path):
        return {"model_name": "DeepFire-VGG19", "error": f"Dataset not found: {dataset_path}", "metrics": None}

    try:
        X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_path)
        vgg_model, _ = build_and_train_vgg19(X_train, y_train, X_test, y_test)

        vgg_probs = vgg_model.predict(X_test).flatten()
        vgg_pred  = (vgg_probs >= 0.5).astype(int)

        stats = evaluate_model('VGG19-TL', y_test, vgg_pred)

        return {
            "model_name": "DeepFire-VGG19",
            "metrics": {
                "accuracy":  float(stats['Accuracy']),
                "precision": float(stats['Precision']),
                "recall":    float(stats['Recall']),
                "f1":        float(stats['F1']),
                "auc":       float(roc_auc_score(y_test, vgg_probs)),
                "aupr":      float(average_precision_score(y_test, vgg_probs)),
            },
        }
    except Exception as exc:
        return {"model_name": "DeepFire-VGG19", "error": str(exc), "metrics": None}


if __name__ == '__main__':
    main()
