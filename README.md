# Sistema de Vision YOLO para Ajedrez

## Breve explicacion del sistema

El sistema permite crear y mejorar un detector de piezas de ajedrez en tres etapas:

1. Captura y etiquetado semiautomatico de imagenes en formato YOLO.
2. Reentrenamiento del modelo con el dataset local.
3. Inferencia en tiempo real usando camara.

Con este flujo se puede iterar rapido: capturar mas muestras de clases con bajo rendimiento, reentrenar y volver a probar en vivo.

## Requisitos

- Python
- Camara USB o integrada
- Piezas de ajedrez
- Dependencias de Python:

```bash
pip install ultralytics opencv-python numpy
```

Opcional (entorno virtual recomendado):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install --upgrade pip
pip install ultralytics opencv-python numpy
```

## Estructura relevante

```text
UT5-Sistema_robotizado/
|- captura_dataset.py
|- reentrenar_modelo.py
|- realtime_inferencia.py
|- dataset/
|  |- data.yaml
|  |- train/
|  |- valid/
|  |- test/
|- runs/detect/
```

## Problemas encontrados

- Instalar las librerias en jetson nano
- Conseguir un dataset con suficientes muestras de cada clase
- Ajustar hiperparametros para mejorar rendimiento
- Detectar piezas similares

## Posibles mejoras futuras

- Aumentar dataset con más imágenes

## Instrucciones de ejecucion

### 1) Captura de dataset ([captura_dataset.py](captura_dataset.py))

Comando base:

```bash
python captura_dataset.py
```

Comando recomendado:

```bash
python captura_dataset.py --camera 0 --box-w 0.24 --box-h 0.58 --valid-ratio 0.15 --burst-size 30 --burst-delay 0.08
```

Controles:

- `ESPACIO`: guardar 1 muestra.
- `b`: rafaga.
- `n` / `p`: cambiar clase.
- `l`: listar clases por consola.
- `0`-`9`: seleccion rapida de clase.
- `q` o `ESC`: salir.

Salida esperada:

- `dataset/train/images` y `dataset/train/labels`
- `dataset/valid/images` y `dataset/valid/labels`

### 2) Reentrenamiento ([reentrenar_modelo.py](reentrenar_modelo.py))

Comando base:

```bash
python reentrenar_modelo.py
```

Comando recomendado:

```bash
python reentrenar_modelo.py --model yolov5n.pt --epochs 60 --imgsz 512 --batch 8 --name chess_nano_v5
```

Fine-tuning desde un modelo ya entrenado:

```bash
python reentrenar_modelo.py --model runs/detect/chess_nano_v5/weights/best.pt --epochs 40 --imgsz 512 --batch 8 --name chess_nano_v5_finetune
```

Resultado:

- `runs/detect/<experimento>/weights/best.pt`
- `runs/detect/<experimento>/weights/last.pt`

### 3) Inferencia en tiempo real ([realtime_inferencia.py](realtime_inferencia.py))

Comando base:

```bash
python realtime_inferencia.py
```

Comando recomendado:

```bash
python realtime_inferencia.py --model runs/detect/chess_nano_v5/weights/best.pt --camera 0 --backend dshow --conf 0.4 --iou 0.45 --imgsz 512
```

Notas:

- Cierra con `q` o `ESC`.
- Si la camara falla, prueba `--camera 1` o `--camera 2`.

## Flujo recomendado

1. Capturar muestras por clase (ideal: 300-500 por clase).
2. Revisar que [dataset/data.yaml](dataset/data.yaml) tenga clases correctas.
3. Reentrenar con 40-60 epocas.
4. Probar inferencia en vivo.
5. Repetir el ciclo para mejorar clases con errores.

## Autores del proyecto

- Eimar Ramos
- Javier Peralta
