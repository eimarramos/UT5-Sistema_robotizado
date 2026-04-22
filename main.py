import cv2
import numpy as np

# Iniciar la cámara
cap = cv2.VideoCapture(0)

# Detector MOG2 (inteligente frente a cambios de luz/fondo)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Aplicar el detector de fondo
    mascara = fgbg.apply(frame)
    # Limpieza de ruido
    _, mascara = cv2.threshold(mascara, 200, 255, cv2.THRESH_BINARY)
    mascara = cv2.dilate(mascara, None, iterations=2)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lógica de detección
    hay_objeto = False
    for contorno in contornos:
        if cv2.contourArea(contorno) > 2000: # Ajusta este valor según el tamaño del objeto
            hay_objeto = True
            (x, y, w, h) = cv2.boundingRect(contorno)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) # Rectángulo rojo si detecta

    # --- INTERFAZ VISUAL ---
    # Dibujar un rectángulo de estado en la parte superior
    color_estado = (0, 0, 255) if hay_objeto else (0, 255, 0) # Rojo si hay, Verde si no
    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1) # Barra negra de fondo
    
    texto = "OBJETO DETECTADO" if hay_objeto else "SIN OBJETO"
    cv2.putText(frame, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_estado, 3)

    # Mostrar ventana
    cv2.imshow("Sistema de Detección", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()