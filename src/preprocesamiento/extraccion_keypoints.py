"""Extraccion de keypoints con MediaPipe Holistic -> vector de 258 features por frame."""
import numpy as np
import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def crear_holistic(min_detection=0.5, min_tracking=0.5):
    return mp_holistic.Holistic(
        min_detection_confidence=min_detection,
        min_tracking_confidence=min_tracking,
    )


def procesar_frame(frame_bgr, holistic):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    resultados = holistic.process(rgb)
    return resultados


def extraer_keypoints(resultados):
    """Concatena pose (132) + mano izq (63) + mano der (63) = 258."""
    pose = (
        np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in resultados.pose_landmarks.landmark]).flatten()
        if resultados.pose_landmarks else np.zeros(33 * 4)
    )
    lh = (
        np.array([[lm.x, lm.y, lm.z] for lm in resultados.left_hand_landmarks.landmark]).flatten()
        if resultados.left_hand_landmarks else np.zeros(21 * 3)
    )
    rh = (
        np.array([[lm.x, lm.y, lm.z] for lm in resultados.right_hand_landmarks.landmark]).flatten()
        if resultados.right_hand_landmarks else np.zeros(21 * 3)
    )
    return np.concatenate([pose, lh, rh])


def dibujar_landmarks(frame, resultados):
    mp_drawing.draw_landmarks(frame, resultados.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, resultados.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, resultados.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    return frame
