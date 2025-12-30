# YOLO Detection App

## Configuración

### Variables de entorno (.env)
```
MODELPATH=/ruta/al/modelo/best12oct.pt
PROCESSINGPATH=/ruta/a/Processing
OUTPUTPATH=/ruta/a/Outputs
OUTPUTFAIL=/ruta/a/Outputs/fail
OUTPUTFILE=/ruta/a/Outputs/detections.txt
```

## Uso

1. **Colocar videos**: Poner el video en la carpeta `Processing/`
2. **Ejecutar**: `python Scripts/main.py`
3. **Resultados**:
   - Video procesado → `Outputs/`
   - Video con error → `Outputs/fail/`
   - Detecciones → `Outputs/detections.txt` (formato: `(frame, cy)`) --> es decir las coordenadas en y en este caso.

## Funcionamiento

- El script revisa cada segundo si hay videos en `Processing/`
- Procesa el primer video encontrado con YOLO (clase 1: rodhead)
- Guarda las coordenadas del centro Y de cada detección
- Mueve el video procesado a `Outputs/` o `Outputs/fail/` según el resultado --> el video original.

