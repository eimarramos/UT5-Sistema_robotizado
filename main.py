import cv2
import numpy as np

# Iniciar la cámara
cap = cv2.VideoCapture(0)

# Detector MOG2
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret: break

    # --- Primer paso: aplicar el detector sin actualizar el fondo si hay objeto ---
    # Primero, aplicamos con learningRate=0 para comprobar si hay objetos
    mascara_temp = fgbg.apply(frame, learningRate=0)
    _, mascara_temp = cv2.threshold(mascara_temp, 200, 255, cv2.THRESH_BINARY)
    mascara_temp = cv2.dilate(mascara_temp, None, iterations=2)

    _, contornos_temp, _ = cv2.findContours(mascara_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hay_objeto = False
    for contorno in contornos_temp:
        if cv2.contourArea(contorno) > 2000: 
            hay_objeto = True
            break  # Si hay al menos uno, ya basta

    # --- Segundo paso: aplicar o no la actualización del fondo ---
    if hay_objeto:
        # No recalibrar el fondo, poner "0 objetos"
        mascara = fgbg.apply(frame, learningRate=0)
        num_objetos = 0
    else:
        # Sí recalibrar el fondo
        mascara = fgbg.apply(frame)
        _, mascara = cv2.threshold(mascara, 200, 255, cv2.THRESH_BINARY)
        mascara = cv2.dilate(mascara, None, iterations=2)
        _, contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_objetos = len([c for c in contornos if cv2.contourArea(c) > 2000])

    # --- Dibujo visual ---
    color_estado = (0, 0, 255) if hay_objeto else (0, 255, 0)
    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1) # Barra negra de fondo

    texto = f"OBJETOS: {num_objetos}" if not hay_objeto else "OBJETO DETECTADO"
    cv2.putText(frame, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_estado, 3)

    # Mostrar ventana
    cv2.imshow("Sistema de Detección", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()