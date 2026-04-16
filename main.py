from ultralytics import YOLO
import cv2

# Cargar modelo
model = YOLO("yolov8n.pt")

# Abrir cámara (usa 0 casi siempre)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error con la cámara")
        break

    # Detección
    results = model(frame)

    # Dibujar cajas y etiquetas
    annotated_frame = results[0].plot()

    # Mostrar resultado
    cv2.imshow("Deteccion YOLO", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()