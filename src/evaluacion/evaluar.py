"""Evaluacion del modelo: accuracy, matriz de confusion, classification report."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from config import CLASSES


def evaluar_modelo(model, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred, target_names=CLASSES))
    return y_pred


def matriz_confusion(y_true, y_pred, guardar=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    plt.colorbar(im)
    plt.tight_layout()
    if guardar:
        plt.savefig(guardar)
    plt.show()
    return cm


def graficar_historial(historial):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(historial.history["loss"], label="train")
    axes[0].plot(historial.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(historial.history["accuracy"], label="train")
    axes[1].plot(historial.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.show()
