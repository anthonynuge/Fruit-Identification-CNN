import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def plot_train_val_accuracy(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_loss'], label='Validation Accuracy')
    plt.title("Accuracy Line Chart")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_train_val_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Loss Line Chart")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_confusion_matrix(model, test_ds, class_names):
    y_true = np.concatenate([y for _, y in test_ds], axis=0)
    y_pred = np.argmax(model.predict(test_ds), axis=1)
    y_true = np.argmax(y_true, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()

def visualize_feature_map(feature_maps, title):
    filters_len = feature_maps.shape[-1]
    plt.figure(figsize=(15,15))
    for i in range(min(filters_len, 16)):
        plt.subplot(4, 4, i + 1)
        plt.imshow(feature_maps[0, :, :, i], cmap='virdis')
        plt.axis("off")
    plt.show()

