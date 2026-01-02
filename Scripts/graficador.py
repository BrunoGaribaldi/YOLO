import ast
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path


# Cargar variables de entorno
def graficar(output_file, run_dir):
    """
    Grafica las detecciones y guarda en /Output/video/grafico_detecciones.png
    
    Args:
        output_file: Path al archivo detections.txt
        run_dir: Directorio base del video (/Output/video)
    """
    # Archivo de salida: /Output/video/grafico_detecciones.png
    out_path = Path(run_dir) / "grafico_detecciones.png"

    # === Leer el archivo ===
    pairs = []
    # Aceptar tanto Path como string
    file_path = Path(output_file) if not isinstance(output_file, Path) else output_file
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                a, b = ast.literal_eval(s)   # Convierte "(1, 407.0)" -> (1, 407.0)
                pairs.append((int(a), float(b)))
            except Exception:
                pass

    # === Crear DataFrame ===
    df = pd.DataFrame(pairs, columns=["frame", "valor"])

    # === Graficar ===
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.scatter(df["frame"], df["valor"], s=8, alpha=0.7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_title("Coordenada Y del centro de detección por frame")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Coordenada Y (cy)")
    ax.grid(True, linewidth=0.5, alpha=0.3)
    plt.margins(x=0.01, y=0.05)
    plt.tight_layout()

    # === Guardar gráfico ===
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Gráfico guardado en {out_path.resolve()}")