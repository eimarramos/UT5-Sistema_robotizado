import cv2
import numpy as np

# ── Configuración ──────────────────────────────────────────────
CAMARA        = 0
ANCHO         = 640
ALTO          = 480
UMBRAL_FG     = 180
HISTORY       = 300
VAR_THRESHOLD = 40
FRAMES_ON     = 4
FRAMES_OFF    = 6
LR_LIBRE      = 0.004
SKIP_FRAMES   = 2

# Recuadro fijo central (porcentaje del frame)
ROI_X1 = int(ANCHO * 0.25)
ROI_Y1 = int(ALTO  * 0.20)
ROI_X2 = int(ANCHO * 0.75)
ROI_Y2 = int(ALTO  * 0.80)

# Área mínima ocupada dentro del ROI para confirmar objeto (%)
FILL_MIN = 0.08   # 8% del ROI debe estar en movimiento
# ──────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(CAMARA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ANCHO)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
cap.set(cv2.CAP_PROP_FPS, 30)

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=HISTORY,
    varThreshold=VAR_THRESHOLD,
    detectShadows=False
)

kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

# Calentar fondo
print("Calibrando fondo...")
for _ in range(30):
    ret, f = cap.read()
    if ret:
        fgbg.apply(f, learningRate=0.1)

# Estado
contador_on  = 0
contador_off = 0
hay_objeto   = False
frame_count  = 0
roi_fill     = 0.0

# Área del ROI en resolución completa
roi_area = (ROI_X2 - ROI_X1) * (ROI_Y2 - ROI_Y1)

print("Listo. Pulsa Q para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    mostrar = frame.copy()

    if frame_count % SKIP_FRAMES == 0:

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gris = cv2.GaussianBlur(gris, (5, 5), 0)

        lr = 0 if hay_objeto else LR_LIBRE
        mascara = fgbg.apply(gris, learningRate=lr)

        _, mascara = cv2.threshold(mascara, UMBRAL_FG, 255, cv2.THRESH_BINARY)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel_open,  iterations=1)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # Recortar máscara al ROI central
        roi_mascara = mascara[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

        # Porcentaje de píxeles activos dentro del ROI
        pixeles_activos = cv2.countNonZero(roi_mascara)
        roi_fill = pixeles_activos / roi_area

        # Suavizado temporal con histeresis
        if roi_fill >= FILL_MIN:
            contador_on  += 1
            contador_off  = 0
        else:
            contador_off += 1
            contador_on   = 0

        if not hay_objeto and contador_on  >= FRAMES_ON:
            hay_objeto = True
        if     hay_objeto and contador_off >= FRAMES_OFF:
            hay_objeto = False

    # ── Dibujar ROI ────────────────────────────────────────────
    color_roi = (0, 50, 220) if hay_objeto else (30, 200, 30)
    grosor_roi = 3 if hay_objeto else 2

    # Esquinas estilo visor
    largo = 30
    for (cx, cy, dx, dy) in [
        (ROI_X1, ROI_Y1,  1,  1),
        (ROI_X2, ROI_Y1, -1,  1),
        (ROI_X1, ROI_Y2,  1, -1),
        (ROI_X2, ROI_Y2, -1, -1),
    ]:
        cv2.line(mostrar, (cx, cy), (cx + dx * largo, cy), color_roi, grosor_roi)
        cv2.line(mostrar, (cx, cy), (cx, cy + dy * largo), color_roi, grosor_roi)

    # Rectángulo central semitransparente
    overlay = mostrar.copy()
    cv2.rectangle(overlay, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), color_roi, -1)
    cv2.addWeighted(overlay, 0.06, mostrar, 0.94, 0, mostrar)
    cv2.rectangle(mostrar, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), color_roi, 1)

    # ── HUD superior ───────────────────────────────────────────
    cv2.rectangle(mostrar, (0, 0), (ANCHO, 50), (10, 10, 10), -1)

    if hay_objeto:
        texto  = "OBJETO DETECTADO"
        color  = (0, 50, 220)
    else:
        texto  = "SIN OBJETO"
        color  = (30, 200, 30)

    cv2.putText(mostrar, texto, (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

    # Barra de llenado del ROI (indicador visual de cuánto movimiento hay)
    barra_max = ANCHO - 160
    barra_val = int(min(roi_fill / FILL_MIN, 1.0) * barra_max)
    cv2.rectangle(mostrar, (ANCHO - barra_max - 10, 60), (ANCHO - 10, 74), (40, 40, 40), -1)
    cv2.rectangle(mostrar, (ANCHO - barra_max - 10, 60), (ANCHO - barra_max - 10 + barra_val, 74), color_roi, -1)
    cv2.putText(mostrar, "MOV", (ANCHO - barra_max - 50, 73),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    cv2.imshow("Deteccion de Objetos", mostrar)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()