# Manual de uso y comandos importantes

Este documento resume el flujo completo:

1. Capturar imagenes con etiquetas YOLO.
2. Limpiar capturas creadas por el programa.
3. Reentrenar el modelo.
4. Probar inferencia en tiempo real.

## 1) Requisitos

Desde la carpeta del proyecto, instala dependencias:

```bash
pip install ultralytics opencv-python
```

## 2) Captura de dataset

Script:

- captura_dataset.py

Comando base:

```bash
python captura_dataset.py
```

Comando con parametros utiles:

```bash
python captura_dataset.py --camera 0 --box-w 0.24 --box-h 0.58 --valid-ratio 0.15 --burst-size 30 --burst-delay 0.08
```

### Controles dentro de la ventana

- ESPACIO: guardar 1 foto con etiqueta
- b: rafaga (30 por defecto)
- n: clase siguiente
- p: clase anterior
- l: listar clases en consola
- 0-9: elegir clase rapida (ids 0..9)
- q o ESC: salir

### Donde se guarda

- dataset/train/images
- dataset/train/labels
- dataset/valid/images
- dataset/valid/labels

### Formato de nombre de captura

Las capturas de este programa se guardan como:

- clase_YYYYMMDD_HHMMSS_NNN.jpg
- clase_YYYYMMDD_HHMMSS_NNN.txt

Ejemplo:

- black-queen_20260506_192417_133.jpg

## 3) Limpieza de capturas creadas por el programa

### Borrar solo capturas de una clase (PowerShell)

Ejemplo para black-queen (solo creadas por el capturador):

```powershell
$base = Join-Path (Get-Location) 'dataset'
$regex = '^(blackqueen|black-queen)_\d{8}_\d{6}_\d{3}\.(jpg|txt)$'
Get-ChildItem -Path $base -Recurse -File |
  Where-Object { $_.Name -match $regex } |
  Remove-Item -Force
```

Ejemplo para white-king:

```powershell
$base = Join-Path (Get-Location) 'dataset'
$regex = '^(whiteking|white-king)_\d{8}_\d{6}_\d{3}\.(jpg|txt)$'
Get-ChildItem -Path $base -Recurse -File |
  Where-Object { $_.Name -match $regex } |
  Remove-Item -Force
```

### Verificar cuantos quedan

```powershell
$base = Join-Path (Get-Location) 'dataset'
$regex = '^(whiteking|white-king)_\d{8}_\d{6}_\d{3}\.(jpg|txt)$'
$left = Get-ChildItem -Path $base -Recurse -File | Where-Object { $_.Name -match $regex }
'Restantes: ' + $left.Count
```

## 4) Reentrenar el modelo

Script:

- reentrenar_modelo.py

Entrenamiento base:

```bash
python reentrenar_modelo.py
```

Entrenamiento recomendado para Jetson Nano (YOLOv5n):

```bash
python reentrenar_modelo.py --model yolov5n.pt --epochs 60 --imgsz 512 --batch 8 --name chess_nano_v5
```

Continuar afinando desde tu ultimo best.pt ya migrado:

```bash
python reentrenar_modelo.py --model runs/detect/chess_nano_v5/weights/best.pt --epochs 40 --imgsz 512 --batch 8 --name chess_nano_v5_finetune
```

Reanudar entrenamiento pausado (desde last.pt):

```bash
yolo detect train resume model=runs/detect/chess_nano_v5/weights/last.pt
```

Nota:

- Usa el last.pt de la carpeta de experimento que quieras continuar.

Resultado esperado:

- runs/detect/chess_nano_v5/weights/best.pt

## 5) Inferencia en tiempo real

Script:

- realtime_inferencia.py

Comando base:

```bash
python realtime_inferencia.py
```

Comando con parametros:

```bash
python realtime_inferencia.py --model runs/detect/chess_nano_v5/weights/best.pt --camera 0 --conf 0.4 --iou 0.45 --imgsz 512
```

Salir de la ventana: q o ESC.

## 6) Flujo recomendado rapido

1. Captura por clase (ideal: 300-500 por clase).
2. Mantener dataset balanceado entre clases.
3. Reentrenar 40-60 epocas.
4. Probar en tiempo real.
5. Si falla en una clase, capturar mas de esa clase y reentrenar.

## 7) Solucion de problemas

- No guarda al presionar ESPACIO:
  - Haz clic en la ventana de OpenCV para que tenga foco.
  - Revisa en consola que aparezca [OK] Guardada ...

- Camara no abre:
  - Prueba otro indice: --camera 1 o --camera 2.

- Entrenamiento muy lento:
  - Baja --imgsz a 512 y/o --batch a 4.

- Clases incorrectas:
  - Verifica nombres en dataset/data.yaml (campo names).

```
python realtime_inferencia.py --model runs/detect/runs/detect/chess_nano_v5/weights/best.pt --camera 0 --backend dshow --conf 0.4 --iou 0.45 --imgsz 512
```
