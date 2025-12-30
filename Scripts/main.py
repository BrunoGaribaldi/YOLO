import detection
import os
import time
import cv2
from dotenv import load_dotenv
import graficador
import bpm
import on_off
# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Obtener la ruta de procesamiento desde variables de entorno
PROCESSING_PATH = os.getenv("PROCESSINGPATH")
SLEEP = 1  # Tiempo de espera entre revisiones (en segundos)

# Loop infinito que revisa si hay archivos para procesar
while True:
    # Listar todos los archivos en la carpeta de procesamiento
    files = os.listdir(PROCESSING_PATH)
    
    # Si hay archivos en la carpeta, procesarlos
    if files:
        # Detección de FPS del video
        video_path = os.path.join(PROCESSING_PATH, files[0])
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f"FPS: {fps}")

        # Crear una instancia de la clase Detection
        detector = detection.Detection()
        
        # Llamar al método detectar() para procesar el video
        detector.detectar()

        #graficamos las detecciones.
        graficador.graficar()

        # Verificar estado ON/OFF del pump jack
        status_result = on_off.check_pump_jack_status()
        if status_result: #si el pump jack esta funcionando, calculamos el BPM.
            print(f"Estado Pump Jack: {status_result['status']} (Confianza: {status_result['confidence']:.0%})")

        
            # Calculamos el BPM
            # === Rutas ===
            # Archivo de entrada desde variables de entorno
            OUTPUT_FILE = os.getenv("OUTPUTFILE")  # detections.txt
            frames, ys = bpm.load_points(OUTPUT_FILE)  # esto es medio redundante, podria traer el dato como salida de la funcion detectar().

            bpm_value, dbg = bpm.bpm_cycle(frames, ys, fps, bpm_min=2, bpm_max=20)
            print("BPM:", bpm_value)
        
        else: #si el pump jack no esta funcionando, no calculamos el BPM.
            print("Pump Jack no esta funcionando")
            print(f"Estado Pump Jack: {status_result['status']} (Confianza: {status_result['confidence']:.0%})")
            continue

    # Esperar antes de revisar nuevamente
    time.sleep(SLEEP)