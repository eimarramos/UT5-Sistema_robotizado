# Sistema de Vision YOLOv5 para Ajedrez

Este proyecto ya no usa el paquete `ultralytics` en Python.
Ahora usa `YOLOv5` clasico (repo oficial local) para entrenamiento e inferencia.

## Flujo general

1. Captura y etiquetado semiautomatico de imagenes en formato YOLO.
2. Reentrenamiento del modelo con dataset local.
3. Inferencia en tiempo real con camara.

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
|- third_party/
|  |- yolov5/        # repo oficial YOLOv5
|- runs/detect/
```

## Requisitos

- Python
- Camara USB o integrada
- Git
- PyTorch compatible con tu hardware
- OpenCV
- PyYAML

## Instalacion

### 1) Entorno virtual (recomendado)

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 2) Clonar YOLOv5 local

```bash
mkdir -p third_party
git clone https://github.com/ultralytics/yolov5.git third_party/yolov5
```

### 3) Instalar dependencias

PC (CUDA o CPU):

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python pyyaml
pip install -r third_party/yolov5/requirements.txt
```

Si usas CPU, cambia el indice de PyTorch por uno compatible CPU.

Jetson Nano:

1. Instala PyTorch/torchvision para tu version exacta de JetPack usando wheels de NVIDIA.
2. Luego instala:

```bash
pip install opencv-python pyyaml
pip install -r third_party/yolov5/requirements.txt
```

Nota: en Nano es comun ajustar versiones de `numpy`/`opencv` segun compatibilidad del sistema.

## Uso

### 1) Captura de dataset

Archivo: `captura_dataset.py`

Comando base:

```bash
python captura_dataset.py
```

Comando recomendado:

```bash
python captura_dataset.py --camera 0 --box-w 0.24 --box-h 0.58 --valid-ratio 0.15 --burst-size 30 --burst-delay 0.08
```

Controles:

- `ESPACIO`: guardar 1 muestra
- `b`: rafaga
- `n` / `p`: cambiar clase
- `l`: listar clases por consola
- `0`-`9`: seleccion rapida de clase
- `q` o `ESC`: salir

Salida:

- `dataset/train/images` y `dataset/train/labels`
- `dataset/valid/images` y `dataset/valid/labels`

### 2) Reentrenamiento YOLOv5

Archivo: `reentrenar_modelo.py`

Comando base:

```bash
python reentrenar_modelo.py
```

Comando recomendado:

```bash
python reentrenar_modelo.py --model yolov5n.pt --epochs 60 --imgsz 512 --batch 8 --name chess_nano_v5 --yolov5-dir third_party/yolov5
```

Fine-tuning desde un modelo previo:

```bash
python reentrenar_modelo.py --model runs/detect/chess_nano_v5/weights/best.pt --epochs 40 --imgsz 512 --batch 8 --name chess_nano_v5_finetune --yolov5-dir third_party/yolov5
```

Notas:

- El script convierte `dataset/data.yaml` a rutas absolutas para evitar errores de path en YOLOv5.
- Resultado esperado:
  - `runs/detect/<experimento>/weights/best.pt`
  - `runs/detect/<experimento>/weights/last.pt`

### 3) Inferencia en tiempo real

Archivo: `realtime_inferencia.py`

Comando base:

```bash
python realtime_inferencia.py --model runs/detect/chess_nano_v5/weights/best.pt --yolov5-dir third_party/yolov5
```

Comando recomendado (Windows webcam):

```bash
python realtime_inferencia.py --model runs/detect/chess_nano_v5/weights/best.pt --camera 0 --backend dshow --conf 0.4 --iou 0.45 --imgsz 512 --yolov5-dir third_party/yolov5
```

Comando recomendado (Jetson Nano USB cam):

```bash
python realtime_inferencia.py --model runs/detect/chess_nano_v5/weights/best.pt --camera 0 --backend any --conf 0.4 --iou 0.45 --imgsz 416 --yolov5-dir third_party/yolov5 --device cuda:0
```

Notas:

- Cerrar con `q` o `ESC`.
- Si falla la camara, prueba `--camera 1` o `--camera 2`.
- En Nano usa `yolov5n` + `imgsz 320/416` para mejor FPS.

## Pruebas de modelo (Windows y Jetson Nano)

### Windows

1. Activar entorno:

```bash
.venv\Scripts\activate
```

2. Probar en camara (tiempo real):

```bash
python realtime_inferencia.py --model runs/detect/chess_nano_v5/weights/best.pt --yolov5-dir third_party/yolov5 --device cuda:0 --camera 0 --backend dshow --imgsz 512
```

3. Medir en conjunto de test:

```bash
python third_party\yolov5\val.py --weights runs/detect/chess_nano_v5/weights/best.pt --data dataset/data.yaml --task test --img 512 --batch 16 --device 0
```

4. Probar deteccion sobre imagenes:

```bash
python third_party\yolov5\detect.py --weights runs/detect/chess_nano_v5/weights/best.pt --source dataset/test/images --img 512 --conf 0.4 --device 0
```

### Jetson Nano

1. Copiar al Nano:

- `runs/detect/.../weights/best.pt`
- `realtime_inferencia.py`
- `third_party/yolov5/`
- opcional: `dataset/data.yaml` y `dataset/test/`

2. Instalar PyTorch/torchvision compatible con la version exacta de JetPack (wheels NVIDIA).

3. Instalar dependencias:

```bash
pip3 install pyyaml
pip3 install -r third_party/yolov5/requirements.txt
```

4. Verificar CUDA:

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

5. Probar en camara:

```bash
python3 realtime_inferencia.py --model /ruta/a/best.pt --yolov5-dir /ruta/a/third_party/yolov5 --device cuda:0 --backend any --imgsz 416 --camera 0
```

6. Para mejorar FPS en Nano:

- usar `--imgsz 320`
- mantener `yolov5n`
- subir `--conf` a `0.5` si necesitas menos falsos positivos

## Flujo recomendado de mejora

1. Capturar muestras por clase (ideal: 300-500 por clase).
2. Verificar clases en `dataset/data.yaml`.
3. Entrenar 40-60 epocas.
4. Probar inferencia en vivo.
5. Repetir ciclo para clases con errores.

## Autores del proyecto

- Eimar Ramos
- Javier Peralta
