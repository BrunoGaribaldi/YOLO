import detection
import os
import time
import cv2
from dotenv import load_dotenv
import graficador
import bpm
import on_off
import shutil
# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Obtener la ruta de procesamiento desde variables de entorno
PROCESSING_PATH = os.getenv("PROCESSINGPATH")
OUTPUT_PATH = os.getenv("OUTPUTPATH")
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

        # Obtener nombre del video
        nombre_video = files[0]
        video_id = os.path.splitext(nombre_video)[0]
        run_dir = os.path.join(OUTPUT_PATH, video_id)  # /Output/video
        
        # Crear una instancia de la clase Detection
        detector = detection.Detection()
        
        # Llamar al método detectar() para procesar el video
        # Retorna: run_dir, output_file
        run_dir_result, output_file = detector.detectar()
        
        if run_dir_result is None:
            # Error en el procesamiento, continuar con siguiente video
            continue
        
        # Usar el run_dir retornado (por si acaso)
        run_dir = run_dir_result

        # Graficar las detecciones: /Output/video/grafico_detecciones.png
        graficador.graficar(output_file, run_dir)

        # Verificar estado ON/OFF del pump jack
        status_result = on_off.check_pump_jack_status(output_file)
        
        # Archivo para guardar resultados BPM y ON/OFF: /Output/video/resultados.txt
        resultados_file = os.path.join(run_dir, "resultados.txt")
        
        if status_result and status_result.get('status') == 'ON':  # Si el pump jack está funcionando, calculamos el BPM
            print(f"Estado Pump Jack: {status_result['status']} (Confianza: {status_result['confidence']:.0%})")

            # Calculamos el BPM
            frames, ys = bpm.load_points(output_file) 
            bpm_value, dbg = bpm.bpm_cycle(frames, ys, fps, bpm_min=0.8, bpm_max=10)
            print("BPM:", bpm_value)
            
            # Guardar resultados en archivo
            with open(resultados_file, "w") as f:
                f.write("=== RESULTADOS DEL ANÁLISIS ===\n\n")
                f.write(f"Estado Pump Jack: {status_result['status']}\n")
                f.write(f"Confianza: {status_result['confidence']:.2%}\n")
                f.write(f"Razón: {status_result['reason']}\n")
                f.write(f"Puntos analizados: {status_result['n_points']}\n\n")
                if bpm_value:
                    f.write(f"BPM: {bpm_value:.2f}\n")
                    f.write(f"Período mediano: {dbg.get('median_period_s', 'N/A')} segundos\n")
                    f.write(f"Número de períodos: {dbg.get('n_periods', 'N/A')}\n")
                else:
                    f.write("BPM: No se pudo calcular\n")
        
        else:  # Si el pump jack no está funcionando o hay error, no calculamos el BPM
            if status_result:
                print("Pump Jack no está funcionando")
                print(f"Estado Pump Jack: {status_result['status']} (Confianza: {status_result['confidence']:.0%})")
                
                # Guardar resultados aunque esté OFF
                with open(resultados_file, "w") as f:
                    f.write("=== RESULTADOS DEL ANÁLISIS ===\n\n")
                    f.write(f"Estado Pump Jack: {status_result['status']}\n")
                    f.write(f"Confianza: {status_result['confidence']:.2%}\n")
                    f.write(f"Razón: {status_result['reason']}\n")
                    f.write(f"Puntos analizados: {status_result['n_points']}\n\n")
                    f.write("BPM: No calculado (Pump Jack apagado)\n")
            else:
                print("Error al verificar estado del Pump Jack")
                with open(resultados_file, "w") as f:
                    f.write("=== RESULTADOS DEL ANÁLISIS ===\n\n")
                    f.write("Error: No se pudo determinar el estado del Pump Jack\n")
            
            continue

    # Esperar antes de revisar nuevamente
    time.sleep(SLEEP)