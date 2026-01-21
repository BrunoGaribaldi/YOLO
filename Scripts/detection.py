# SE ENCARGA DE LA DETECCION DE LOS RODHEADS.
import os
import shutil
import torch
from dotenv import load_dotenv
from ultralytics import YOLO
import cv2

class Detection:
    def __init__(self):
        load_dotenv()
        self.MODEL_PATH = os.getenv("MODELPATH")
        self.PROCESSING_PATH = os.getenv("PROCESSINGPATH")
        self.OUTPUT_PATH = os.getenv("OUTPUTPATH")      # raíz /outputs
        self.OUTPUT_FAIL = os.getenv("OUTPUTFAIL")
        self.COCOPATH = os.getenv("COCOPATH")

    def detectar(self):
        nombre_video = os.listdir(self.PROCESSING_PATH)[0]
        video_path = os.path.join(self.PROCESSING_PATH, nombre_video)

        # Extraer nombre sin extensión para crear carpeta
        video_id = os.path.splitext(nombre_video)[0]
        run_dir = os.path.join(self.OUTPUT_PATH, video_id)  # /Output/video
        os.makedirs(run_dir, exist_ok=True)

        # Ruta para detecciones: /Output/video/detections.txt
        output_file = os.path.join(run_dir, "detections.txt")

        # Verificar disponibilidad de GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Advertencia: GPU no disponible, usando CPU (será más lento)")

        model = YOLO(self.MODEL_PATH)
        # Mover modelo a GPU si está disponible
        if device == 'cuda':
            model.to(device)

        coco = YOLO(self.COCOPATH)

        if device == 'cuda':
            coco.to(device)


        try:
            centros = []  # [(frame, cx, cy)]

             # Preparar writer del video final combinado
            yolo_dir = os.path.join(run_dir, "yolo")
            os.makedirs(yolo_dir, exist_ok=True)

            video_yolo_salida = os.path.join(yolo_dir, f"{video_id}.avi")

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(video_yolo_salida, fourcc, fps, (w, h))

            # Procesar video con tu YOLO en stream, SIN guardar automáticamente
            for frame_idx, r_custom in enumerate(
                model(
                    video_path,
                    stream=True,
                    save=False,         # <- clave: no guardar el video "solo rodhead"
                    classes=[0, 1],   # detecta ambas clases. 1 es  rodhead y 0 es aib
                    device=device
                ),
                start=1
            ):
                # Frame original
                frame = r_custom.orig_img  # numpy BGR

                # 1) Registrar SOLO rodhead (class_id=1) en detections.txt
                if r_custom.boxes is not None and len(r_custom.boxes) > 0:
                    for box in r_custom.boxes:
                        if int(box.cls[0]) != 1:
                            continue

                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        centros.append((frame_idx, cx, cy))
                        break  # si querés solo 1 rodhead por frame


                # 2) Correr COCO SOLO para dibujar (NO escribir detecciones)
                # COCO ids típicos: 0=person, 2=car, 7=truck
                r_coco = coco.predict(
                    frame,
                    conf=0.35,
                    classes=[0, 2, 7],
                    device=device,
                    verbose=False
                )[0]

                # 3) Dibujar ambas salidas sobre el mismo frame
                frame_anno = r_custom.plot()             # dibuja rodhead
                frame_anno = r_coco.plot(img=frame_anno) # dibuja COCO encima

                out.write(frame_anno)

            out.release()

            # Escribir detecciones: SOLO rodhead, tal cual pediste
            with open(output_file, "w") as f:
                for frame, cx, cy in centros:
                    f.write(f"({frame}, {cy})\n")

            # Mover video original a: /Output/video/video.mp4
            video_destino = os.path.join(run_dir, nombre_video)
            shutil.move(video_path, video_destino)

            print(f"Video procesado exitosamente: {nombre_video}")
            print(f"  - Video YOLO combinado: {video_yolo_salida}")
            print(f"  - Video original: {video_destino}")
            print(f"  - Detecciones (solo rodhead): {output_file}")
            return run_dir, output_file

        except Exception as e:
            print("Error procesando video:", e)
            shutil.move(video_path, os.path.join(self.OUTPUT_FAIL, nombre_video))
            print(f"Video movido a carpeta de fallidos: {nombre_video}")
            return None, None