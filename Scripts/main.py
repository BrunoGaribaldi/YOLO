import detection
import os
import time
import cv2
from dotenv import load_dotenv
import graficador
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
        #graficador.graficar()
    
    # Esperar antes de revisar nuevamente
    time.sleep(SLEEP)





