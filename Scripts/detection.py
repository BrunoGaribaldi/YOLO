#SE ENCARGA DE LA DETECCION DE LOS RODHEADS.
import os
import shutil
from dotenv import load_dotenv
from ultralytics import YOLO

# Cargar variables de entorno desde el archivo .env
class Detection:
    def __init__(self):

        load_dotenv()
        # Obtener rutas desde variables de entorno
        self.MODEL_PATH = os.getenv("MODELPATH")
        self.PROCESSING_PATH = os.getenv("PROCESSINGPATH")
        self.OUTPUT_PATH = os.getenv("OUTPUTPATH")
        self.OUTPUT_FAIL = os.getenv("OUTPUTFAIL")
        self.OUTPUT_FILE = os.getenv("OUTPUTFILE")


# Obtener el primer archivo de la carpeta de procesamiento
# Nota: esto asume que solo hay un video en la carpeta

    def detectar(self):
        
        archivos = os.listdir(self.PROCESSING_PATH)
        if not archivos:
            raise FileNotFoundError("No se encontraron archivos en la carpeta de procesamiento")

        nombre_video = archivos[0]
        video_path = os.path.join(self.PROCESSING_PATH, nombre_video)

        # Cargar el modelo YOLO entrenado
        model = YOLO(self.MODEL_PATH)

        try:
            # Procesar el video con YOLO
            # stream=True procesa frame por frame (más eficiente en memoria)
            # Cada iteración = un frame del video
            # classes=[1] filtra para detectar solo la clase 1 (rodhead) --> para saber el numero de la clase, se puede ver en el archivo best12oct.pt con print(model.names)
            centros = []  # [(frame, cx, cy)] centro en c y centro en y. A nosotros nos interesa mas que nada en y xq el movimiento relevante es en y, pero no cuesta nada guardar ambos.

            for frame_idx, r in enumerate(
                model(
                    video_path,
                    stream=True,
                    save=True,
                    project=self.OUTPUT_PATH,
                    name="yolo",
                    exist_ok=True,
                    classes=[1]
                    #device=0  #forzamos uso gpu
                ),
                start=1
            ):
                if len(r.boxes) == 0:
                    continue  # sin detección en este frame. Podriamos poner None.

                # tomamos la primera detección (asumís 1 rodhead)
                x1, y1, x2, y2 = map(float, r.boxes[0].xyxy[0])

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                centros.append((frame_idx, cx, cy))
                print((frame_idx, cx, cy))

            # Si el procesamiento fue exitoso, mover el video original a la carpeta de salida
            shutil.move(
                video_path,
                os.path.join(self.OUTPUT_PATH, nombre_video)
            )

            #Escribimos archivo de salida, con detecciones usando y. detections.txt
            with open(self.OUTPUT_FILE, "w") as f:
                for frame, cx, cy in centros:
                    f.write(f"({frame}, {cy})\n")  # usando Y como separador.

            print(f"Video procesado exitosamente: {nombre_video}")

        except Exception as e:
            # Si ocurre un error durante el procesamiento
            print("Error procesando video:", e)

            # Mover el video fallido a la carpeta de videos fallidos
            shutil.move(
                video_path,
                os.path.join(self.OUTPUT_FAIL, nombre_video)
            )
            print(f"Video movido a carpeta de fallidos: {nombre_video}")
