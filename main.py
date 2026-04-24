import cv2
import numpy as np

# Iniciar cámara
cap = cv2.VideoCapture(0)

# Detector MOG2
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Parámetros
AREA_MIN = 2500
FRAMES_OBJETO = 5
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Estado
contador_objeto = 0
hay_objeto = False

# Calentar el modelo de fondo con los primeros frames
for _ in range(30):
    ret, frame = cap.read()
    if ret:
        fgbg.apply(frame, learningRate=0.1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Paso 1: detectar si hay objeto SIN actualizar fondo ---
    mascara_check = fgbg.apply(frame, learningRate=0)
    _, mascara_check = cv2.threshold(mascara_check, 200, 255, cv2.THRESH_BINARY)
    mascara_check = cv2.morphologyEx(mascara_check, cv2.MORPH_OPEN, kernel, iterations=2)
    mascara_check = cv2.dilate(mascara_check, kernel, iterations=3)

    _, contornos_check, _ = cv2.findContours(mascara_check, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grandes_check = [c for c in contornos_check if cv2.contourArea(c) > AREA_MIN]
    hay_objeto_frame = len(grandes_check) > 0

    # --- Paso 2: suavizado temporal ---
    if hay_objeto_frame:
        contador_objeto = min(contador_objeto + 1, FRAMES_OBJETO * 2)
    else:
        contador_objeto = max(contador_objeto - 1, 0)

    hay_objeto = contador_objeto >= FRAMES_OBJETO

    # --- Paso 3: actualizar fondo solo cuando NO hay objeto ---
    if not hay_objeto:
        fgbg.apply(frame, learningRate=0.005)

    # --- Paso 4: calcular máscara final y contornos para dibujar ---
    mascara_final = fgbg.apply(frame, learningRate=0)
    _, mascara_final = cv2.threshold(mascara_final, 200, 255, cv2.THRESH_BINARY)
    mascara_final = cv2.morphologyEx(mascara_final, cv2.MORPH_OPEN, kernel, iterations=2)
    mascara_final = cv2.dilate(mascara_final, kernel, iterations=3)

    _, contornos_final, _ = cv2.findContours(mascara_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objetos = [c for c in contornos_final if cv2.contourArea(c) > AREA_MIN]
    num_objetos = len(objetos) if hay_objeto else 0

    # --- Paso 5: dibujar rectángulos ---
    for contorno in objetos:
        if hay_objeto:
            (x, y, w, h) = cv2.boundingRect(contorno)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # --- Paso 6: interfaz visual ---
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)

    if hay_objeto:
        texto = f"OBJETOS DETECTADOS: {num_objetos}"
        color = (0, 0, 255)
    else:
        texto = "OBJETOS: 0"
        color = (0, 255, 0)

    cv2.putText(frame, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow("Sistema de Deteccion", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()