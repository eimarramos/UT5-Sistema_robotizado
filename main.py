import cv2
import numpy as np

# ── Configuración ──────────────────────────────────────────────
CAMARA        = 0
ANCHO         = 1280
ALTO          = 720
SKIP_FRAMES   = 2          # procesar 1 de cada N frames
FRAMES_ON     = 5          # frames seguidos para confirmar presencia
FRAMES_OFF    = 8          # frames seguidos para confirmar ausencia
DIFF_UMBRAL   = 25         # diferencia de píxel para considerar cambio
FILL_MIN      = 0.02       # % mínimo del ROI alterado para detectar objeto
ALPHA_FONDO   = 0.002      # velocidad de actualización del fondo (muy lenta)

# ROI central
ROI_X1 = int(ANCHO * 0.25)
ROI_Y1 = int(ALTO  * 0.20)
ROI_X2 = int(ANCHO * 0.75)
ROI_Y2 = int(ALTO  * 0.80)
# ──────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(CAMARA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ANCHO)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
cap.set(cv2.CAP_PROP_FPS, 30)

roi_area = (ROI_X2 - ROI_X1) * (ROI_Y2 - ROI_Y1)

kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# ── Aprender fondo inicial ─────────────────────────────────────
print("Calibrando fondo (mantén el recuadro vacío)...")
muestras = []
while len(muestras) < 40:
    ret, f = cap.read()
    if not ret:
        continue
    roi = cv2.cvtColor(f[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2], cv2.COLOR_BGR2GRAY)
    roi = cv2.GaussianBlur(roi, (7, 7), 0)
    muestras.append(roi.astype(np.float32))

    # Mostrar progreso
    prog = f.copy()
    pct  = int(len(muestras) / 40 * (ROI_X2 - ROI_X1))
    cv2.rectangle(prog, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (30, 200, 30), 2)
    cv2.rectangle(prog, (ROI_X1, ROI_Y2 + 8), (ROI_X1 + pct, ROI_Y2 + 22), (30, 200, 30), -1)
    cv2.putText(prog, "Calibrando fondo...", (ROI_X1, ROI_Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 200, 30), 1)
    cv2.imshow("Deteccion de Objetos", prog)
    cv2.waitKey(1)

# Fondo inicial = mediana de las muestras (robusto a ruido)
fondo = np.median(muestras, axis=0).astype(np.float32)

# Estado
contador_on  = 0
contador_off = 0
hay_objeto   = False
frame_count  = 0
roi_fill     = 0.0

print("Listo. Pulsa Q para salir, R para recalibrar fondo.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    mostrar = frame.copy()

    if frame_count % SKIP_FRAMES == 0:

        # Extraer ROI en gris y suavizado
        roi_gris = cv2.cvtColor(
            frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2],
            cv2.COLOR_BGR2GRAY
        ).astype(np.float32)
        roi_gris = cv2.GaussianBlur(roi_gris, (7, 7), 0)

        # Diferencia absoluta con el fondo de referencia
        diff = cv2.absdiff(roi_gris, fondo)
        _, mascara = cv2.threshold(diff, DIFF_UMBRAL, 255, cv2.THRESH_BINARY)
        mascara = mascara.astype(np.uint8)

        # Morfología: eliminar ruido y rellenar huecos
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel_open,  iterations=1)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # Porcentaje de ROI alterado
        roi_fill = cv2.countNonZero(mascara) / roi_area

        # Actualizar fondo MUY lentamente solo si NO hay objeto
        # → se adapta a cambios de iluminación pero no absorbe objetos estáticos
        if not hay_objeto:
            cv2.accumulateWeighted(roi_gris, fondo, ALPHA_FONDO)

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
    grosor    = 3 if hay_objeto else 2

    # Fondo semitransparente del ROI
    overlay = mostrar.copy()
    cv2.rectangle(overlay, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), color_roi, -1)
    cv2.addWeighted(overlay, 0.06, mostrar, 0.94, 0, mostrar)

    # Esquinas estilo visor
    L = 28
    for (cx, cy, dx, dy) in [
        (ROI_X1, ROI_Y1,  1,  1),
        (ROI_X2, ROI_Y1, -1,  1),
        (ROI_X1, ROI_Y2,  1, -1),
        (ROI_X2, ROI_Y2, -1, -1),
    ]:
        cv2.line(mostrar, (cx, cy), (cx + dx * L, cy), color_roi, grosor)
        cv2.line(mostrar, (cx, cy), (cx, cy + dy * L), color_roi, grosor)

    # ── HUD superior ───────────────────────────────────────────
    cv2.rectangle(mostrar, (0, 0), (ANCHO, 50), (10, 10, 10), -1)
    texto = "OBJETO DETECTADO" if hay_objeto else "SIN OBJETO"
    cv2.putText(mostrar, texto, (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color_roi, 2)

    # Barra de diferencia respecto al fondo
    barra_w   = 180
    barra_val = int(min(roi_fill / FILL_MIN, 1.0) * barra_w)
    bx        = ANCHO - barra_w - 14
    cv2.rectangle(mostrar, (bx, 58), (bx + barra_w, 72), (40, 40, 40), -1)
    cv2.rectangle(mostrar, (bx, 58), (bx + barra_val, 72), color_roi, -1)
    cv2.putText(mostrar, "DIFF", (bx - 42, 71),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    cv2.imshow("Deteccion de Objetos", mostrar)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('q'):
        break
    elif tecla == ord('r'):
        # Recalibrar fondo en caliente
        print("Recalibrando...")
        muestras = []
        while len(muestras) < 40:
            ret, f = cap.read()
            if not ret:
                continue
            roi = cv2.cvtColor(f[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2], cv2.COLOR_BGR2GRAY)
            roi = cv2.GaussianBlur(roi, (7, 7), 0)
            muestras.append(roi.astype(np.float32))
            cv2.waitKey(1)
        fondo = np.median(muestras, axis=0).astype(np.float32)
        hay_objeto   = False
        contador_on  = 0
        contador_off = 0
        print("Fondo recalibrado.")

cap.release()
cv2.destroyAllWindows()