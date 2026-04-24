import cv2
import numpy as np

# Iniciar cámara
cap = cv2.VideoCapture(0)

# Detector MOG2
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Parámetros
AREA_MIN = 2500          # Área mínima para considerar objeto
FRAMES_OBJETO = 5        # Número de frames consecutivos para confirmar detección
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# Contador de frames con objeto
contador_objeto = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    # Aplicar fondo sin actualizar para verificar objetos
    mascara_temp = fgbg.apply(frame, learningRate=0)
    _, mascara_temp = cv2.threshold(mascara_temp, 200, 255, cv2.THRESH_BINARY)
    # Limpiar ruido: apertura (erosión + dilatación)
    mascara_temp = cv2.morphologyEx(mascara_temp, cv2.MORPH_OPEN, kernel, iterations=1)
    mascara_temp = cv2.dilate(mascara_temp, kernel, iterations=2)

    # Contornos temporales
    _, contornos_temp, _ = cv2.findContours(mascara_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Comprobar si hay objeto grande
    hay_objeto_frame = any(cv2.contourArea(c) > AREA_MIN for c in contornos_temp)

    # Suavizado temporal
    if hay_objeto_frame:
        contador_objeto += 1
    else:
        contador_objeto = max(contador_objeto - 1, 0)

    hay_objeto = contador_objeto >= FRAMES_OBJETO

    # Aplicar actualización de fondo solo si no hay objeto
    if hay_objeto:
        mascara = mascara_temp
        contornos = contornos_temp
        num_objetos = 0
    else:
        mascara = fgbg.apply(frame)
        _, mascara = cv2.threshold(mascara, 200, 255, cv2.THRESH_BINARY)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)
        mascara = cv2.dilate(mascara, kernel, iterations=2)
        _, contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_objetos = len([c for c in contornos if cv2.contourArea(c) > AREA_MIN])

    # Dibujar rectángulos sobre objetos visibles
    for contorno in contornos:
        if cv2.contourArea(contorno) > AREA_MIN:
            (x, y, w, h) = cv2.boundingRect(contorno)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Interfaz visual
    color_estado = (0, 0, 255) if hay_objeto else (0, 255, 0)
    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
    texto = "OBJETO DETECTADO" if hay_objeto else f"OBJETOS: {num_objetos}"
    cv2.putText(frame, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_estado, 3)

    cv2.imshow("Sistema de Detección", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()