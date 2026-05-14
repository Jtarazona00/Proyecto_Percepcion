"""Entrypoint: orquesta baseline, entrenamiento, evaluacion e inferencia en tiempo real."""
import argparse

from src.utils.datos import cargar_dataset, particionar
from src.modelos.baseline_svm import features_dataset, entrenar_svm
from src.entrenamiento.entrenar import entrenar
from src.evaluacion.evaluar import evaluar_modelo, matriz_confusion, graficar_historial
from src.inferencia.tiempo_real import inferir_tiempo_real


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "modo",
        choices=["baseline", "entrenar", "evaluar", "tiempo_real"],
    )
    args = parser.parse_args()

    if args.modo == "tiempo_real":
        inferir_tiempo_real()
        return

    X, y = cargar_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = particionar(X, y)

    if args.modo == "baseline":
        X_train_stats = features_dataset(X_train)
        X_test_stats = features_dataset(X_test)
        entrenar_svm(X_train_stats, y_train, X_test_stats, y_test)
        return

    if args.modo == "entrenar":
        model, historial = entrenar(X_train, y_train, X_val, y_val)
        graficar_historial(historial)
        y_pred = evaluar_modelo(model, X_test, y_test)
        matriz_confusion(y_test, y_pred)
        return

    if args.modo == "evaluar":
        import tensorflow as tf
        from config import MODEL_PATH
        model = tf.keras.models.load_model(MODEL_PATH)
        y_pred = evaluar_modelo(model, X_test, y_test)
        matriz_confusion(y_test, y_pred)


if __name__ == "__main__":
    main()
