# SE ENCARGA DE LA DETECCION DE LOS RODHEADS.
import os
import shutil
from dotenv import load_dotenv
from ultralytics import YOLO

class Detection:
    def __init__(self):
        load_dotenv()
        self.MODEL_PATH = os.getenv("MODELPATH")
        self.PROCESSING_PATH = os.getenv("PROCESSINGPATH")
        self.OUTPUT_PATH = os.getenv("OUTPUTPATH")      # raíz /outputs
        self.OUTPUT_FAIL = os.getenv("OUTPUTFAIL")

    def detectar(self):
        nombre_video = os.listdir(self.PROCESSING_PATH)[0]
        video_path = os.path.join(self.PROCESSING_PATH, nombre_video)

        # Extraer nombre sin extensión para crear carpeta
        video_id = os.path.splitext(nombre_video)[0]
        run_dir = os.path.join(self.OUTPUT_PATH, video_id)  # /Output/video
        os.makedirs(run_dir, exist_ok=True)

        # Ruta para detecciones: /Output/video/detections.txt
        output_file = os.path.join(run_dir, "detections.txt")

        model = YOLO(self.MODEL_PATH)

        try:
            centros = []  # [(frame, cx, cy)]

            # Procesar video con YOLO
            # El resultado se guarda en: /Output/video/yolo/video.avi
            for frame_idx, r in enumerate(
                model(
                    video_path,
                    stream=True,
                    save=True,
                    project=run_dir,   # /Output/video
                    name="yolo",       # /Output/video/yolo
                    exist_ok=True,
                    classes=[1]
                ),
                start=1
            ):
                if len(r.boxes) == 0:
                    continue

                x1, y1, x2, y2 = map(float, r.boxes[0].xyxy[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                centros.append((frame_idx, cx, cy))

            # Escribir detecciones: /Output/video/detections.txt
            with open(output_file, "w") as f:
                for frame, cx, cy in centros:
                    f.write(f"({frame}, {cy})\n")

            # YOLO guarda el video procesado en: /Output/video/yolo/video.avi
            # (usa el nombre del archivo original pero con extensión .avi)
            yolo_dir = os.path.join(run_dir, "yolo")
            video_id = os.path.splitext(nombre_video)[0]
            video_yolo_esperado = os.path.join(yolo_dir, f"{video_id}.avi")
            
            # Si YOLO guardó con otro nombre, renombrarlo
            if os.path.exists(yolo_dir):
                archivos_yolo = [f for f in os.listdir(yolo_dir) if f.endswith(('.avi', '.mp4'))]
                if archivos_yolo and archivos_yolo[0] != os.path.basename(video_yolo_esperado):
                    # Renombrar al nombre esperado
                    archivo_actual = os.path.join(yolo_dir, archivos_yolo[0])
                    if not os.path.exists(video_yolo_esperado):
                        shutil.move(archivo_actual, video_yolo_esperado)

            # Mover video original a: /Output/video/video.mp4
            video_destino = os.path.join(run_dir, nombre_video)
            shutil.move(video_path, video_destino)

            print(f"Video procesado exitosamente: {nombre_video}")
            print(f"  - Video YOLO: {video_yolo_esperado}")
            print(f"  - Video original: {video_destino}")
            print(f"  - Detecciones: {output_file}")
            return run_dir, output_file  # Retornar rutas para uso en main.py

        except Exception as e:
            print("Error procesando video:", e)
            shutil.move(video_path, os.path.join(self.OUTPUT_FAIL, nombre_video))
            print(f"Video movido a carpeta de fallidos: {nombre_video}")
            return None, None
