import cv2
import numpy as np

# ── Configuración ──────────────────────────────────────────────
CAMARA        = 0
ANCHO         = 640
ALTO          = 480
AREA_MIN      = 1500      # área mínima de contorno (px²)
UMBRAL_FG     = 180       # umbral para binarizar la máscara
HISTORY       = 300
VAR_THRESHOLD = 40
FRAMES_ON     = 4         # frames seguidos para confirmar presencia
FRAMES_OFF    = 6         # frames seguidos para confirmar ausencia
LR_LIBRE      = 0.004     # learning rate sin objeto
SKIP_FRAMES   = 2         # procesar 1 de cada N frames
# ──────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(CAMARA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ANCHO)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
cap.set(cv2.CAP_PROP_FPS, 30)

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=HISTORY,
    varThreshold=VAR_THRESHOLD,
    detectShadows=False        # detectShadows=False → mucho más rápido
)

kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

# Calentar modelo de fondo (sin mostrar ventana)
print("Calibrando fondo...")
for _ in range(25):
    ret, f = cap.read()
    if ret:
        fgbg.apply(cv2.resize(f, (ANCHO // 2, ALTO // 2)), learningRate=0.1)

# Estado
contador_on  = 0
contador_off = 0
hay_objeto   = False
num_objetos  = 0
bboxes       = []
frame_count  = 0

print("Detectando... (pulsa Q para salir)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    mostrar = frame.copy()

    # ── Procesar solo cada SKIP_FRAMES frames ──────────────────
    if frame_count % SKIP_FRAMES == 0:

        # Reducir resolución para procesar más rápido
        pequeño = cv2.resize(frame, (ANCHO // 2, ALTO // 2))
        gris    = cv2.cvtColor(pequeño, cv2.COLOR_BGR2GRAY)
        gris    = cv2.GaussianBlur(gris, (5, 5), 0)

        # Aplicar sustractor (sin actualizar fondo si hay objeto)
        lr = 0 if hay_objeto else LR_LIBRE
        mascara = fgbg.apply(gris, learningRate=lr)

        # Binarizar (sombras ya desactivadas, pero por si acaso)
        _, mascara = cv2.threshold(mascara, UMBRAL_FG, 255, cv2.THRESH_BINARY)

        # Morfología: quitar ruido y rellenar huecos
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel_open,  iterations=1)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # Contornos en imagen pequeña
        _, contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar por área (área en escala pequeña → AREA_MIN/4)
        grandes = [c for c in contornos if cv2.contourArea(c) > AREA_MIN / 4]

        # ── Suavizado temporal con histeresis ─────────────────
        if grandes:
            contador_on  += 1
            contador_off  = 0
        else:
            contador_off += 1
            contador_on   = 0

        if not hay_objeto and contador_on  >= FRAMES_ON:
            hay_objeto = True
        if     hay_objeto and contador_off >= FRAMES_OFF:
            hay_objeto = False

        # Escalar bboxes a resolución original (×2)
        if hay_objeto:
            num_objetos = len(grandes)
            bboxes = []
            for c in grandes:
                x, y, w, h = cv2.boundingRect(c)
                bboxes.append((x * 2, y * 2, w * 2, h * 2))
        else:
            num_objetos = 0
            bboxes = []

    # ── Dibujar resultados ─────────────────────────────────────
    for (x, y, w, h) in bboxes:
        cv2.rectangle(mostrar, (x, y), (x + w, y + h), (0, 60, 255), 2)
        cv2.putText(mostrar, "OBJ", (x + 4, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 60, 255), 1)

    # HUD
    color_hud = (0, 50, 200) if hay_objeto else (30, 160, 30)
    cv2.rectangle(mostrar, (0, 0), (ANCHO, 50), (10, 10, 10), -1)
    texto = f"OBJETOS DETECTADOS: {num_objetos}" if hay_objeto else "OBJETOS: 0"
    cv2.putText(mostrar, texto, (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_hud, 2)

    cv2.imshow("Deteccion de Objetos", mostrar)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()