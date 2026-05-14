"""Baseline clasico: SVM sobre features estadisticas por landmark."""
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from config import SCALER_PATH, SVM_PATH


def extraer_caracteristicas_estadisticas(secuencia):
    # secuencia: array de forma (30, 258)
    media = np.mean(secuencia, axis=0)
    std   = np.std(secuencia, axis=0)
    rango = np.max(secuencia, axis=0) - np.min(secuencia, axis=0)
    return np.concatenate([media, std, rango])  # vector de 774 features


def features_dataset(secuencias):
    return np.array([extraer_caracteristicas_estadisticas(s) for s in secuencias])


def entrenar_svm(X_train_stats, y_train, X_test_stats, y_test):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_stats)
    X_test_sc  = scaler.transform(X_test_stats)

    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    svm.fit(X_train_sc, y_train)
    precision_baseline = accuracy_score(y_test, svm.predict(X_test_sc))
    print(f'Precisión Baseline SVM: {precision_baseline:.2%}')  # ~67.3%

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(svm, SVM_PATH)
    return svm, scaler, precision_baseline
